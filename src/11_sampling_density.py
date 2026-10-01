"""
Sampling-density control experiment.

This measures the error of the raw point-cloud UDF at different point counts.
It is an input-quality control, not a full VAD ablation yet.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

grid_size = 220
axis = np.linspace(-1.5, 1.5, grid_size)
xx, yy = np.meshgrid(axis, axis)
queries = np.column_stack((xx.ravel(), yy.ravel()))
true_udf = np.abs(np.sqrt(xx**2 + yy**2) - 1.0)

point_counts = [10, 20, 40, 80, 160]
results = {}
mean_errors = []
max_errors = []

for count in point_counts:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))

    distances = np.linalg.norm(
        queries[:, None, :] - points[None, :, :], axis=2
    )
    udf = distances.min(axis=1).reshape(grid_size, grid_size)
    error = np.abs(udf - true_udf)

    results[count] = (points, udf)
    mean_errors.append(error.mean())
    max_errors.append(error.max())

    print(
        f"{count:3d} points | "
        f"mean error = {error.mean():.6f} | "
        f"max error = {error.max():.6f}"
    )

fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

for ax, count in zip(axes[:3], [10, 40, 160]):
    points, udf = results[count]
    image = ax.contourf(xx, yy, udf, levels=60, cmap="viridis")
    ax.scatter(points[:, 0], points[:, 1], s=8, c="white")
    ax.set_title(f"Raw point-cloud UDF: {count} points")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax)

axes[3].plot(point_counts, mean_errors, "o-", label="mean absolute error")
axes[3].plot(point_counts, max_errors, "s--", label="maximum error")
axes[3].set_xscale("log", base=2)
axes[3].set_xlabel("number of input points")
axes[3].set_ylabel("UDF error")
axes[3].set_title("Sampling-density control")
axes[3].grid(alpha=0.3)
axes[3].legend()

fig.suptitle("VAD Stage 10: Effect of input point density")
fig.tight_layout()

output_path = Path(__file__).resolve().parents[1] / "outputs" / "11_sampling_density.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")