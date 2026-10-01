import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output_path = root / "outputs" / "18_noisy_ellipse_baseline.png"

rng = np.random.default_rng(21)

# 干净椭圆
a, b = 1.20, 0.72
n_points = 80
theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
clean_points = np.column_stack([a * np.cos(theta), b * np.sin(theta)])

# 加入各向同性高斯位置噪声
noise_sigma = 0.035
noisy_points = clean_points + rng.normal(
    loc=0.0,
    scale=noise_sigma,
    size=clean_points.shape,
)

# 查询网格
axis = np.linspace(-1.7, 1.7, 180)
xx, yy = np.meshgrid(axis, axis)
grid = np.column_stack([xx.ravel(), yy.ravel()])

def nearest_distance(query, cloud):
    return np.linalg.norm(
        query[:, None, :] - cloud[None, :, :],
        axis=2,
    ).min(axis=1)

# 噪声点的离散 UDF
noisy_udf = nearest_distance(grid, noisy_points).reshape(xx.shape)

# 高密度干净椭圆作为参考真值
theta_ref = np.linspace(0, 2 * np.pi, 16000, endpoint=False)
reference_points = np.column_stack([
    a * np.cos(theta_ref),
    b * np.sin(theta_ref),
])
reference_udf = nearest_distance(grid, reference_points).reshape(xx.shape)

error = np.abs(noisy_udf - reference_udf)
point_rmse = np.sqrt(np.mean(np.sum((noisy_points - clean_points) ** 2, axis=1)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

axes[0].scatter(
    clean_points[:, 0], clean_points[:, 1],
    c="tab:green", s=20, label="clean reference",
)
axes[0].scatter(
    noisy_points[:, 0], noisy_points[:, 1],
    c="crimson", s=18, label="noisy input",
)
for clean, noisy in zip(clean_points, noisy_points):
    axes[0].plot(
        [clean[0], noisy[0]],
        [clean[1], noisy[1]],
        color="gray",
        alpha=0.45,
        lw=0.7,
    )
axes[0].set_title("Controlled noisy input points")
axes[0].set_aspect("equal")
axes[0].set_xlim(-1.7, 1.7)
axes[0].set_ylim(-1.7, 1.7)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()

im1 = axes[1].imshow(
    noisy_udf,
    extent=[-1.7, 1.7, -1.7, 1.7],
    origin="lower",
    cmap="viridis",
)
axes[1].scatter(noisy_points[:, 0], noisy_points[:, 1], c="white", s=8)
axes[1].set_title("Raw UDF from noisy points")
axes[1].set_aspect("equal")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(
    error,
    extent=[-1.7, 1.7, -1.7, 1.7],
    origin="lower",
    cmap="magma",
)
axes[2].scatter(noisy_points[:, 0], noisy_points[:, 1], c="white", s=6)
axes[2].set_title("Error against clean ellipse UDF")
axes[2].set_aspect("equal")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
fig.colorbar(im2, ax=axes[2])

fig.suptitle("Noise baseline before VAD position optimisation", y=1.02)
fig.tight_layout()
fig.savefig(output_path, dpi=180, bbox_inches="tight")

np.savez(
    root / "outputs" / "18_noisy_ellipse_input.npz",
    clean_points=clean_points,
    noisy_points=noisy_points,
    noise_sigma=noise_sigma,
    reference_points=reference_points,
)

print(f"Gaussian noise sigma: {noise_sigma:.6f}")
print(f"Point-position RMSE: {point_rmse:.6f}")
print(f"Raw noisy-UDF mean absolute error: {error.mean():.6f}")
print(f"Saved figure to: {output_path}")