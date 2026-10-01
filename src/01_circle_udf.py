"""
Stage 0: a verifiable 2D UDF baseline.

This is not the VAD normal-optimization algorithm yet.
It checks our point-cloud sampling, grid coordinates, UDF definition,
and figure output before implementing the paper's modules.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----- 1. Input point cloud P: 200 points sampled on a unit circle -----
num_points = 200
radius = 1.0
angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
points = radius * np.column_stack((np.cos(angles), np.sin(angles)))


# ----- 2. Query grid X: positions where we evaluate the field u(x) -----
grid_size = 300
axis = np.linspace(-1.5, 1.5, grid_size)
xx, yy = np.meshgrid(axis, axis)
queries = np.column_stack((xx.ravel(), yy.ravel()))


# ----- 3. Discrete UDF: distance from x to its nearest sampled point -----
# u_P(x) = min_{p_i in P} ||x - p_i||
distances = np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=2)
udf_from_points = distances.min(axis=1).reshape(grid_size, grid_size)


# Ground truth for a perfect continuous circle:
# u(x) = | ||x|| - radius |
udf_ground_truth = np.abs(np.sqrt(xx**2 + yy**2) - radius)
absolute_error = np.abs(udf_from_points - udf_ground_truth)

print(f"Point-cloud UDF mean absolute error: {absolute_error.mean():.6f}")
print(f"Point-cloud UDF maximum error:      {absolute_error.max():.6f}")


# ----- 4. Visualisation -----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

axes[0].scatter(points[:, 0], points[:, 1], s=10, c="black")
axes[0].set_title("Input: unoriented point cloud")
axes[0].set_aspect("equal")
axes[0].set_xlim(-1.5, 1.5)
axes[0].set_ylim(-1.5, 1.5)

im1 = axes[1].contourf(xx, yy, udf_from_points, levels=60, cmap="viridis")
axes[1].scatter(points[:, 0], points[:, 1], s=3, c="white")
axes[1].set_title(r"Discrete UDF: $u_P(x)=\min_i \|x-p_i\|$")
axes[1].set_aspect("equal")
fig.colorbar(im1, ax=axes[1], label="distance")

im2 = axes[2].imshow(
    absolute_error,
    extent=[-1.5, 1.5, -1.5, 1.5],
    origin="lower",
    cmap="magma",
)
axes[2].set_title("Absolute error vs. true circle UDF")
axes[2].set_aspect("equal")
fig.colorbar(im2, ax=axes[2], label="absolute error")

fig.suptitle("VAD reproduction - Stage 0: UDF sanity check", fontsize=14)
fig.tight_layout()

output_path = Path(__file__).resolve().parents[1] / "outputs" / "01_circle_udf.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")