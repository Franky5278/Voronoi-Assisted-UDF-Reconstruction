"""
Stage 3: Voronoi diagram.
VAD evaluates its normal-alignment energy only on these bisectors.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

num_points = 40
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
points = np.column_stack((np.cos(angles), np.sin(angles)))

vor = Voronoi(points)

fig, ax = plt.subplots(figsize=(7, 7))
voronoi_plot_2d(
    vor, ax=ax,
    show_points=False,
    show_vertices=False,
    line_colors="royalblue",
    line_width=1.2,
    line_alpha=0.8,
)

ax.scatter(points[:, 0], points[:, 1], s=20, c="black", zorder=3)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")
ax.set_title("Stage 3: Voronoi bisectors between point-cloud sites")
ax.grid(alpha=0.2)

output_path = Path(__file__).resolve().parents[1] / "outputs" / "04_voronoi_bisectors.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")