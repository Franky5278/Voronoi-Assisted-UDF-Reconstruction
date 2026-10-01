import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output_path = root / "outputs" / "13_ellipse_baseline.png"

# 椭圆：x^2/a^2 + y^2/b^2 = 1
a, b = 1.20, 0.72

# 输入点云：只保留较少的椭圆采样点
theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
points = np.column_stack([a * np.cos(theta), b * np.sin(theta)])

# 高密度参考曲线：近似“真实椭圆”，用于计算基准 UDF
theta_ref = np.linspace(0, 2 * np.pi, 12000, endpoint=False)
reference = np.column_stack([a * np.cos(theta_ref), b * np.sin(theta_ref)])

# 查询网格
axis = np.linspace(-1.7, 1.7, 180)
xx, yy = np.meshgrid(axis, axis)
query = np.column_stack([xx.ravel(), yy.ravel()])

def nearest_distance(query_points, cloud):
    dist = np.linalg.norm(
        query_points[:, None, :] - cloud[None, :, :],
        axis=2,
    )
    return dist.min(axis=1)

# 80 个输入点构成的离散 UDF
udf_sparse = nearest_distance(query, points).reshape(xx.shape)

# 12000 个参考点近似的真实椭圆 UDF
udf_reference = nearest_distance(query, reference).reshape(xx.shape)

error = np.abs(udf_sparse - udf_reference)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

im0 = axes[0].imshow(
    udf_sparse, extent=[-1.7, 1.7, -1.7, 1.7],
    origin="lower", cmap="viridis",
)
axes[0].scatter(points[:, 0], points[:, 1], c="white", s=10)
axes[0].set_title("Input point-cloud UDF: 80 ellipse points")
axes[0].set_aspect("equal")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    udf_reference, extent=[-1.7, 1.7, -1.7, 1.7],
    origin="lower", cmap="viridis",
)
axes[1].plot(reference[:, 0], reference[:, 1], "w--", lw=1.2)
axes[1].set_title("Dense-reference UDF of the ellipse")
axes[1].set_aspect("equal")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(
    error, extent=[-1.7, 1.7, -1.7, 1.7],
    origin="lower", cmap="magma",
)
axes[2].scatter(points[:, 0], points[:, 1], c="white", s=8)
axes[2].set_title("Absolute sampling error")
axes[2].set_aspect("equal")
fig.colorbar(im2, ax=axes[2])

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.suptitle("VAD test input: ellipse instead of a circle", y=1.02)
fig.tight_layout()
fig.savefig(output_path, dpi=180, bbox_inches="tight")

print(f"Mean absolute UDF error: {error.mean():.6f}")
print(f"Maximum UDF error: {error.max():.6f}")
print(f"Saved figure to: {output_path}")