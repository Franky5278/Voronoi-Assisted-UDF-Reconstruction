import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

root = Path(__file__).resolve().parents[1]
output_path = root / "outputs" / "17_paper_udf_integration.png"

# 读取论文方程版的融合梯度场 Y_f
data = np.load(root / "outputs" / "16_paper_diffused_fields.npz")
xx = data["xx"]
yy = data["yy"]
fused = data["fused_field"]
points = data["points"]

H, W = xx.shape
axis = xx[0]
grid_h = axis[1] - axis[0]

# ------------------------------------------------------------
# 论文 Eq.(14)/(15)：
# min_u ∫ ||∇u - Y_f||²，且 u(p_i)=0
# 其 Euler-Lagrange 方程就是 Δu = div(Y_f)
# ------------------------------------------------------------
rows, cols, values, rhs = [], [], [], []
row = 0

def index(i, j):
    return i * W + j

# 水平方向：u[i,j+1] - u[i,j] ≈ h * Y_f,x
for i in range(H):
    for j in range(W - 1):
        rows += [row, row]
        cols += [index(i, j), index(i, j + 1)]
        values += [-1.0, 1.0]
        rhs.append(grid_h * fused[i, j, 0])
        row += 1

# 竖直方向：u[i+1,j] - u[i,j] ≈ h * Y_f,y
for i in range(H - 1):
    for j in range(W):
        rows += [row, row]
        cols += [index(i, j), index(i + 1, j)]
        values += [-1.0, 1.0]
        rhs.append(grid_h * fused[i, j, 1])
        row += 1

# Dirichlet 条件：所有输入点都是 UDF 零水平集
constraint_weight = 80.0
for point in points:
    j = np.argmin(np.abs(axis - point[0]))
    i = np.argmin(np.abs(axis - point[1]))

    rows.append(row)
    cols.append(index(i, j))
    values.append(constraint_weight)
    rhs.append(0.0)
    row += 1

A = coo_matrix(
    (values, (rows, cols)),
    shape=(row, H * W),
).tocsr()

u_raw = lsqr(
    A,
    np.asarray(rhs),
    atol=1e-8,
    btol=1e-8,
    iter_lim=3000,
)[0].reshape(H, W)

# 理想 UDF 应非负；保留原始最小值作诊断，同时展示非负 UDF
udf = np.maximum(u_raw, 0.0)

# ------------------------------------------------------------
# 高密度椭圆参考 UDF
# ------------------------------------------------------------
a, b = 1.20, 0.72
theta_ref = np.linspace(0, 2 * np.pi, 16000, endpoint=False)
reference_points = np.column_stack([
    a * np.cos(theta_ref),
    b * np.sin(theta_ref),
])

grid = np.column_stack([xx.ravel(), yy.ravel()])
reference_udf = np.linalg.norm(
    grid[:, None, :] - reference_points[None, :, :],
    axis=2,
).min(axis=1).reshape(H, W)

error = np.abs(udf - reference_udf)

# ------------------------------------------------------------
# 可视化
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))

im0 = axes[0].imshow(
    udf,
    extent=[axis.min(), axis.max(), axis.min(), axis.max()],
    origin="lower",
    cmap="viridis",
)
axes[0].contour(xx, yy, udf, levels=[0.025], colors="white", linewidths=1.3)
axes[0].scatter(points[:, 0], points[:, 1], c="white", s=8)
axes[0].set_title("Recovered UDF from paper equations")
axes[0].set_aspect("equal")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    reference_udf,
    extent=[axis.min(), axis.max(), axis.min(), axis.max()],
    origin="lower",
    cmap="viridis",
)
axes[1].contour(xx, yy, reference_udf, levels=[0.025], colors="white", linewidths=1.3)
axes[1].set_title("Dense-reference ellipse UDF")
axes[1].set_aspect("equal")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(
    error,
    extent=[axis.min(), axis.max(), axis.min(), axis.max()],
    origin="lower",
    cmap="magma",
)
axes[2].scatter(points[:, 0], points[:, 1], c="white", s=6)
axes[2].set_title("Absolute reconstruction error")
axes[2].set_aspect("equal")
fig.colorbar(im2, ax=axes[2])

# y=0 截面，直观看两条 UDF 曲线是否一致
middle = H // 2
axes[3].plot(axis, udf[middle], label="recovered", lw=2)
axes[3].plot(axis, reference_udf[middle], "--", label="reference", lw=2)
axes[3].set_title("Horizontal UDF cross-section (y=0)")
axes[3].set_xlabel("x")
axes[3].set_ylabel("UDF value")
axes[3].grid(alpha=0.25)
axes[3].legend()

for ax in axes[:3]:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.suptitle(
    r"Paper-equation UDF recovery: $\Delta u=\nabla\cdot Y_f$",
    y=1.02,
)
fig.tight_layout()
fig.savefig(output_path, dpi=180, bbox_inches="tight")

np.savez(
    root / "outputs" / "17_paper_ellipse_udf.npz",
    udf=udf,
    raw_solution=u_raw,
    reference_udf=reference_udf,
    error=error,
    xx=xx,
    yy=yy,
)

print(f"Raw Poisson solution minimum: {u_raw.min():.8f}")
print(f"Mean absolute UDF error: {error.mean():.8f}")
print(f"Maximum UDF error: {error.max():.8f}")
print(f"Saved figure to: {output_path}")