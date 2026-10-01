"""
Stage 2: VAD projection distance field.

Paper Eq. (1):
d_{p_i, ~v_i}(x) = |(x - p_i) dot v_i| / ||v_i||

Definition 2:
Use the vector belonging to x's nearest point-cloud site.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

num_points = 80
radius = 1.0
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
points = radius * np.column_stack((np.cos(angles), np.sin(angles)))

# Controlled reference only: true radial normal of the circle.
true_vectors = points / np.linalg.norm(points, axis=1, keepdims=True)

# This represents VAD's random initial bidirectional vectors.
rng = np.random.default_rng(7)
random_angles = rng.uniform(0, 2 * np.pi, num_points)
random_vectors = np.column_stack((np.cos(random_angles), np.sin(random_angles)))

grid_size = 300
axis = np.linspace(-1.5, 1.5, grid_size)
xx, yy = np.meshgrid(axis, axis)
queries = np.column_stack((xx.ravel(), yy.ravel()))

# Nearest point p_j assigns the Voronoi cell containing each query x.
euclidean_distances = np.linalg.norm(
    queries[:, None, :] - points[None, :, :], axis=2
)
nearest_indices = euclidean_distances.argmin(axis=1)

def projection_field(vectors):
    """Evaluate VAD Eq. (1) using the nearest site's vector."""
    nearest_points = points[nearest_indices]
    nearest_vectors = vectors[nearest_indices]
    offsets = queries - nearest_points

    numerator = np.abs(np.sum(offsets * nearest_vectors, axis=1))
    denominator = np.linalg.norm(nearest_vectors, axis=1)
    return (numerator / denominator).reshape(grid_size, grid_size)

field_true = projection_field(true_vectors)
field_random = projection_field(random_vectors)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, field, title in [
    (axes[0], field_true, "Correct bidirectional normals"),
    (axes[1], field_random, "Random initial bidirectional vectors"),
]:
    image = ax.contourf(xx, yy, field, levels=60, cmap="viridis")
    ax.scatter(points[:, 0], points[:, 1], s=3, c="white")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="projection distance")

fig.suptitle("VAD Stage 2: Projection distance field")
fig.tight_layout()

output_path = Path(__file__).resolve().parents[1] / "outputs" / "03_projection_field.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")