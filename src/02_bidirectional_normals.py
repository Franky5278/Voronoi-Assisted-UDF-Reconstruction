"""
Stage 1: visualise bi-directional normals.

For an unoriented point p_i, VAD keeps a PAIR {n_i, -n_i},
rather than deciding which side is globally "outside".
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Same input point cloud: a unit circle.
num_points = 80
radius = 1.0
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
points = radius * np.column_stack((np.cos(angles), np.sin(angles)))

# For this controlled toy circle only, we know the geometric normal.
# In a real VAD input, this direction is unknown and must be optimised.
n = points / np.linalg.norm(points, axis=1, keepdims=True)
minus_n = -n

# Draw fewer arrows so the figure remains readable.
show = np.arange(0, num_points, 4)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(points[:, 0], points[:, 1], s=12, c="black", label="unoriented points")

# Red and blue arrows are the two equally valid directions at each point.
ax.quiver(
    points[show, 0], points[show, 1],
    n[show, 0], n[show, 1],
    color="crimson", scale=12, width=0.006, label=r"$+n_i$"
)
ax.quiver(
    points[show, 0], points[show, 1],
    minus_n[show, 0], minus_n[show, 1],
    color="royalblue", scale=12, width=0.006, label=r"$-n_i$"
)

ax.set_aspect("equal")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title(r"Stage 1: bidirectional normal pair $\{n_i, -n_i\}$")
ax.legend(loc="upper right")
ax.grid(alpha=0.25)

output_path = Path(__file__).resolve().parents[1] / "outputs" / "02_bidirectional_normals.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")