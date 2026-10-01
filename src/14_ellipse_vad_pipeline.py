import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

root = Path(__file__).resolve().parents[1]
output_path = root / "outputs" / "14_ellipse_vad_pipeline.png"

# ------------------------------------------------------------
# 1. 无向椭圆点云
# ------------------------------------------------------------
rng = np.random.default_rng(7)
a, b = 1.20, 0.72
n_points = 80
theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
points = np.column_stack([a * np.cos(theta), b * np.sin(theta)])

# 随机初始化双向法向轴：每个点只知道 {n, -n}
angles = rng.uniform(0, 2 * np.pi, n_points)
vectors = np.column_stack([np.cos(angles), np.sin(angles)])

# ------------------------------------------------------------
# 2. 简化的 VAD 法向轴优化
# 邻点连线近似局部切线，因此法向轴应与该连线正交
# ------------------------------------------------------------
losses = []
for _ in range(900):
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    tangent = following - previous
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    # 把每根轴向量投影到“垂直局部切线”的方向
    normal_component = vectors - np.sum(vectors * tangent, axis=1, keepdims=True) * tangent
    normal_component /= np.linalg.norm(normal_component, axis=1, keepdims=True)

    # 小步更新，模拟优化过程
    vectors = 0.985 * vectors + 0.015 * normal_component
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    align_error = np.mean(np.sum(vectors * tangent, axis=1) ** 2)
    losses.append(align_error)

# ------------------------------------------------------------
# 3. 网格；构造张量场 T=n⊗n 和有向向量场
# ------------------------------------------------------------
axis = np.linspace(-1.7, 1.7, 161)
xx, yy = np.meshgrid(axis, axis)
grid = np.column_stack([xx.ravel(), yy.ravel()])
h = axis[1] - axis[0]

# 每个网格位置的最近输入点
dist_to_points = np.linalg.norm(grid[:, None, :] - points[None, :, :], axis=2)
nearest = np.argmin(dist_to_points, axis=1)
n_near = vectors[nearest]

# 张量场（n 和 -n 完全等价）
Txx = (n_near[:, 0] * n_near[:, 0]).reshape(xx.shape)
Txy = (n_near[:, 0] * n_near[:, 1]).reshape(xx.shape)
Tyy = (n_near[:, 1] * n_near[:, 1]).reshape(xx.shape)

# 张量扩散
sigma = 2.2
Txx_s = gaussian_filter(Txx, sigma=sigma)
Txy_s = gaussian_filter(Txy, sigma=sigma)
Tyy_s = gaussian_filter(Tyy, sigma=sigma)

# 提取平滑张量的主轴；符号仍然不确定
trace = Txx_s + Tyy_s
delta = np.sqrt((Txx_s - Tyy_s) ** 2 + 4 * Txy_s ** 2)
lam = 0.5 * (trace + delta)
axis_x = Txy_s
axis_y = lam - Txx_s
length = np.sqrt(axis_x ** 2 + axis_y ** 2) + 1e-12
axis_x /= length
axis_y /= length

# ------------------------------------------------------------
# 4. 通过 p±εn 建立向量场，再扩散；恢复方向
# ------------------------------------------------------------
eps = 0.06
source_plus = points + eps * vectors
source_minus = points - eps * vectors
sources = np.vstack([source_plus, source_minus])
source_vectors = np.vstack([vectors, -vectors])

dist_to_sources = np.linalg.norm(
    grid[:, None, :] - sources[None, :, :],
    axis=2,
)
nearest_source = np.argmin(dist_to_sources, axis=1)
vx = source_vectors[nearest_source, 0].reshape(xx.shape)
vy = source_vectors[nearest_source, 1].reshape(xx.shape)

vx_s = gaussian_filter(vx, sigma=sigma)
vy_s = gaussian_filter(vy, sigma=sigma)

# 用扩散向量场为张量主轴确定正负号，然后融合
dot = axis_x * vx_s + axis_y * vy_s
sign = np.where(dot >= 0, 1.0, -1.0)
fused_x = sign * axis_x
fused_y = sign * axis_y

# ------------------------------------------------------------
# 5. 最小二乘积分：寻找 UDF，使 ∇u ≈ fused field
#    同时强制点云位置 u=0
# ------------------------------------------------------------
H, W = xx.shape
rows, cols, data, rhs = [], [], [], []
row = 0

def idx(i, j):
    return i * W + j

# x 方向差分：u(i,j+1)-u(i,j) ≈ h * F_x
for i in range(H):
    for j in range(W - 1):
        rows += [row, row]
        cols += [idx(i, j), idx(i, j + 1)]
        data += [-1.0, 1.0]
        rhs.append(h * fused_x[i, j])
        row += 1

# y 方向差分：u(i+1,j)-u(i,j) ≈ h * F_y
for i in range(H - 1):
    for j in range(W):
        rows += [row, row]
        cols += [idx(i, j), idx(i + 1, j)]
        data += [-1.0, 1.0]
        rhs.append(h * fused_y[i, j])
        row += 1

# 点云约束：UDF 在采样点处应为 0
weight = 30.0
for point in points:
    j = np.argmin(np.abs(axis - point[0]))
    i = np.argmin(np.abs(axis - point[1]))
    rows.append(row)
    cols.append(idx(i, j))
    data.append(weight)
    rhs.append(0.0)
    row += 1

A = coo_matrix((data, (rows, cols)), shape=(row, H * W)).tocsr()
udf = lsqr(A, np.asarray(rhs), atol=1e-7, btol=1e-7)[0].reshape(H, W)
udf -= udf.min()

# ------------------------------------------------------------
# 6. 高密度参考椭圆与误差
# ------------------------------------------------------------
theta_ref = np.linspace(0, 2 * np.pi, 12000, endpoint=False)
reference = np.column_stack([a * np.cos(theta_ref), b * np.sin(theta_ref)])

distance_reference = np.linalg.norm(
    grid[:, None, :] - reference[None, :, :],
    axis=2,
).min(axis=1).reshape(H, W)

error = np.abs(udf - distance_reference)

# ------------------------------------------------------------
# 7. 可视化
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].scatter(points[:, 0], points[:, 1], c="black", s=12)
axes[0, 0].quiver(
    points[:, 0], points[:, 1],
    vectors[:, 0], vectors[:, 1],
    color="crimson", scale=16,
)
axes[0, 0].quiver(
    points[:, 0], points[:, 1],
    -vectors[:, 0], -vectors[:, 1],
    color="royalblue", scale=16,
)
axes[0, 0].set_title("Optimised bidirectional normal axes")
axes[0, 0].set_aspect("equal")
axes[0, 0].set_xlim(-1.7, 1.7)
axes[0, 0].set_ylim(-1.7, 1.7)

step = 8
axes[0, 1].quiver(
    xx[::step, ::step], yy[::step, ::step],
    axis_x[::step, ::step], axis_y[::step, ::step],
    color="royalblue",
)
axes[0, 1].scatter(points[:, 0], points[:, 1], c="black", s=8)
axes[0, 1].set_title("Tensor-diffused principal axes")
axes[0, 1].set_aspect("equal")

axes[0, 2].quiver(
    xx[::step, ::step], yy[::step, ::step],
    fused_x[::step, ::step], fused_y[::step, ::step],
    color="crimson",
)
axes[0, 2].scatter(points[:, 0], points[:, 1], c="black", s=8)
axes[0, 2].set_title("Fused signed vector field")
axes[0, 2].set_aspect("equal")

im0 = axes[1, 0].imshow(
    udf, extent=[-1.7, 1.7, -1.7, 1.7],
    origin="lower", cmap="viridis",
)
axes[1, 0].contour(xx, yy, udf, levels=[0.035], colors="white")
axes[1, 0].scatter(points[:, 0], points[:, 1], c="white", s=7)
axes[1, 0].set_title("Recovered UDF")
axes[1, 0].set_aspect("equal")
fig.colorbar(im0, ax=axes[1, 0])

im1 = axes[1, 1].imshow(
    distance_reference, extent=[-1.7, 1.7, -1.7, 1.7],
    origin="lower", cmap="viridis",
)
axes[1, 1].contour(xx, yy, distance_reference, levels=[0.035], colors="white")
axes[1, 1].set_title("Dense-reference UDF")
axes[1, 1].set_aspect("equal")
fig.colorbar(im1, ax=axes[1, 1])

im2 = axes[1, 2].imshow(
    error, extent=[-1.7, 1.7, -1.7, 1.7],
    origin="lower", cmap="magma",
)
axes[1, 2].set_title("Absolute reconstruction error")
axes[1, 2].set_aspect("equal")
fig.colorbar(im2, ax=axes[1, 2])

for ax in axes.ravel():
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.suptitle("VAD-style pipeline on an ellipse", y=0.98)
fig.tight_layout()
fig.savefig(output_path, dpi=180, bbox_inches="tight")

print(f"Final normal-axis loss: {losses[-1]:.8f}")
print(f"Mean absolute UDF error: {error.mean():.6f}")
print(f"Maximum UDF error: {error.max():.6f}")
print(f"Saved figure to: {output_path}")