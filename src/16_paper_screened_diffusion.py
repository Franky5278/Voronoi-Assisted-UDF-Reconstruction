import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

root = Path(__file__).resolve().parents[1]
output_path = root / "outputs" / "16_paper_screened_diffusion.png"

# ------------------------------------------------------------
# 读取第 15 步：论文 Voronoi 能量优化得到的无向法向轴
# ------------------------------------------------------------
data = np.load(root / "outputs" / "15_paper_optimised_normals.npz")
points = data["points"]
normals = data["vectors"]

# 论文尺度：h 为平均最近邻距离，t=h²，epsilon=1e-4 h
pairwise_dist = np.linalg.norm(
    points[:, None, :] - points[None, :, :],
    axis=2,
)
pairwise_dist[pairwise_dist == 0] = np.inf
h_nn = pairwise_dist.min(axis=1).mean()
t = h_nn ** 2
epsilon_paper = 1e-4 * h_nn

# ------------------------------------------------------------
# 2D 计算域与网格
# ------------------------------------------------------------
grid_size = 161
axis = np.linspace(-1.7, 1.7, grid_size)
xx, yy = np.meshgrid(axis, axis)
H, W = xx.shape
grid_h = axis[1] - axis[0]

# epsilon_paper 远小于网格间距，p+epsilon*n 和 p-epsilon*n
# 会在栅格化时几乎抵消。因此只为“向量点源离散化”采用可分辨偏移。
# 论文的 t=h²、screened-Poisson 方程、tensor/vector/fusion 流程不变。
epsilon_grid = max(epsilon_paper, 1.5 * grid_h)

# ------------------------------------------------------------
# 离散求解论文方程：
# (Delta - 1/t)Y = -S/t
# 等价为： (I - t Delta)Y = S
# ------------------------------------------------------------
def index(i, j):
    return i * W + j

rows, cols, values = [], [], []

for i in range(H):
    for j in range(W):
        center = index(i, j)
        neighbours = []

        if i > 0:
            neighbours.append(index(i - 1, j))
        if i < H - 1:
            neighbours.append(index(i + 1, j))
        if j > 0:
            neighbours.append(index(i, j - 1))
        if j < W - 1:
            neighbours.append(index(i, j + 1))

        degree = len(neighbours)

        # I - t*Delta，其中 Delta 使用 5 点差分
        rows.append(center)
        cols.append(center)
        values.append(1.0 + t * degree / grid_h**2)

        for neighbour in neighbours:
            rows.append(center)
            cols.append(neighbour)
            values.append(-t / grid_h**2)

A = coo_matrix(
    (values, (rows, cols)),
    shape=(H * W, H * W),
).tocsc()

# 同一个线性算子需要解多个右端项，只分解一次
solver = splu(A)

# ------------------------------------------------------------
# 连续点源的双线性栅格化
# ------------------------------------------------------------
def splat(field, position, value):
    x, y = position

    gx = (x - axis[0]) / grid_h
    gy = (y - axis[0]) / grid_h

    j0 = int(np.floor(gx))
    i0 = int(np.floor(gy))

    dx = gx - j0
    dy = gy - i0

    contributions = [
        (i0,     j0,     (1.0 - dx) * (1.0 - dy)),
        (i0,     j0 + 1, (1.0 - dx) * dx),
        (i0 + 1, j0,     dy * (1.0 - dx)),
        (i0 + 1, j0 + 1, dy * dx),
    ]

    for i, j, weight in contributions:
        if 0 <= i < H and 0 <= j < W:
            field[i, j] += weight * value

# ------------------------------------------------------------
# 论文 Tensor Diffusion：
# T=n⊗n，再解 (I-tΔ)Y_t=T
# ------------------------------------------------------------
Txx = np.zeros((H, W))
Txy = np.zeros((H, W))
Tyy = np.zeros((H, W))

for p, n in zip(points, normals):
    splat(Txx, p, n[0] * n[0])
    splat(Txy, p, n[0] * n[1])
    splat(Tyy, p, n[1] * n[1])

Yt_xx = solver.solve(Txx.ravel()).reshape(H, W)
Yt_xy = solver.solve(Txy.ravel()).reshape(H, W)
Yt_yy = solver.solve(Tyy.ravel()).reshape(H, W)

tensor_field = np.zeros((H, W, 2, 2))
tensor_field[..., 0, 0] = Yt_xx
tensor_field[..., 0, 1] = Yt_xy
tensor_field[..., 1, 0] = Yt_xy
tensor_field[..., 1, 1] = Yt_yy

# 最大特征值对应的主轴：保留正交方向，但不含正负号
_, eigenvectors = np.linalg.eigh(tensor_field)
principal_axes = eigenvectors[..., :, 1]

# ------------------------------------------------------------
# 论文 Vector Diffusion：
# p±epsilon*n 分别携带 ±n，再解 (I-tΔ)Y_v=N
# ------------------------------------------------------------
Nx = np.zeros((H, W))
Ny = np.zeros((H, W))

for p, n in zip(points, normals):
    plus_position = p + epsilon_grid * n
    minus_position = p - epsilon_grid * n

    splat(Nx, plus_position, n[0])
    splat(Ny, plus_position, n[1])

    splat(Nx, minus_position, -n[0])
    splat(Ny, minus_position, -n[1])

Yv_x = solver.solve(Nx.ravel()).reshape(H, W)
Yv_y = solver.solve(Ny.ravel()).reshape(H, W)
vector_field = np.stack([Yv_x, Yv_y], axis=-1)

# ------------------------------------------------------------
# 论文 Field Fusion：
# 若 Y_t 主轴和 Y_v 夹角 > 90°，翻转主轴
# ------------------------------------------------------------
dot_product = np.sum(principal_axes * vector_field, axis=-1)

fused_field = principal_axes * np.where(
    dot_product[..., None] >= 0.0,
    1.0,
    -1.0,
)

# 保存，供第 17 步 Poisson UDF 恢复读取
np.savez(
    root / "outputs" / "16_paper_diffused_fields.npz",
    xx=xx,
    yy=yy,
    tensor_field=tensor_field,
    vector_field=vector_field,
    fused_field=fused_field,
    points=points,
    normals=normals,
    h_nn=h_nn,
    t=t,
    epsilon_paper=epsilon_paper,
    epsilon_used_on_grid=epsilon_grid,
)

# ------------------------------------------------------------
# 可视化
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
step = 8

panels = [
    (principal_axes, "Tensor diffusion: principal axes", "royalblue"),
    (vector_field, "Vector diffusion: signed guidance", "crimson"),
    (fused_field, r"Field fusion: $Y_f \approx \nabla u$", "crimson"),
]

for ax, (field, title, colour) in zip(axes, panels):
    shown = field[::step, ::step]
    shown_norm = np.linalg.norm(shown, axis=-1, keepdims=True)
    shown_unit = shown / np.maximum(shown_norm, 1e-12)

    ax.quiver(
        xx[::step, ::step],
        yy[::step, ::step],
        shown_unit[..., 0],
        shown_unit[..., 1],
        color=colour,
        scale=22,
        width=0.004,
    )
    ax.scatter(points[:, 0], points[:, 1], c="black", s=10)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.suptitle(
    r"Paper-equation diffusion: $(\Delta - 1/t)Y=-S/t$",
    y=1.02,
)
fig.tight_layout()
fig.savefig(output_path, dpi=180, bbox_inches="tight")

print(f"Mean nearest-neighbour distance h: {h_nn:.10f}")
print(f"Diffusion time t=h^2: {t:.10e}")
print(f"Paper epsilon=1e-4*h: {epsilon_paper:.10e}")
print(f"Grid-resolved epsilon used: {epsilon_grid:.10e}")
print(
    "Mean vector-field magnitude: "
    f"{np.linalg.norm(vector_field, axis=-1).mean():.10e}"
)
print(f"Saved figure to: {output_path}")