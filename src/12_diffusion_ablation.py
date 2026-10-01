"""
Ablation: no diffusion vs. full VAD diffusion/fusion.

Both methods use the same optimised point normals.
Only the field-construction stage is different.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

root = Path(__file__).resolve().parents[1]

normal_data = np.load(root / "outputs" / "06_optimised_normals.npz")
fused_data = np.load(root / "outputs" / "09_fused_field.npz")

points = normal_data["points"]
normals = normal_data["vectors"]
full_field = fused_data["field"]
xx, yy = fused_data["xx"], fused_data["yy"]

height, width = xx.shape
spacing = xx[0, 1] - xx[0, 0]
queries = np.column_stack((xx.ravel(), yy.ravel()))

# Baseline: no diffusion. Each location directly uses its nearest
# perturbed normal-copy direction.
epsilon = 0.06
source_positions = np.concatenate(
    (points + epsilon * normals, points - epsilon * normals), axis=0
)
source_vectors = np.concatenate((normals, -normals), axis=0)

distances = np.linalg.norm(
    queries[:, None, :] - source_positions[None, :, :], axis=2
)
nearest = distances.argmin(axis=1)
no_diffusion_field = source_vectors[nearest].reshape(height, width, 2)

# Same point constraints u(p_i)=0 for both methods.
constraint = np.zeros((height, width), dtype=bool)
for px, py in points:
    col = int(np.clip(round((px - xx[0, 0]) / spacing), 0, width - 1))
    row = int(np.clip(round((py - yy[0, 0]) / spacing), 0, height - 1))
    constraint[row, col] = True

unknown = ~constraint
index = -np.ones((height, width), dtype=int)
index[unknown] = np.arange(unknown.sum())

def integrate_gradient(field):
    """Least-squares Poisson/gradient integration with fixed zero samples."""
    rows, cols, values, rhs = [], [], [], []
    equation = 0

    def add_edge(ar, ac, br, bc, target):
        nonlocal equation
        a_unknown = unknown[ar, ac]
        b_unknown = unknown[br, bc]

        if not a_unknown and not b_unknown:
            return
        if a_unknown:
            rows.append(equation)
            cols.append(index[ar, ac])
            values.append(-1.0)
        if b_unknown:
            rows.append(equation)
            cols.append(index[br, bc])
            values.append(1.0)
        rhs.append(target)
        equation += 1

    for row in range(height):
        for col in range(width - 1):
            target = spacing * 0.5 * (field[row, col, 0] + field[row, col + 1, 0])
            add_edge(row, col, row, col + 1, target)

    for row in range(height - 1):
        for col in range(width):
            target = spacing * 0.5 * (field[row, col, 1] + field[row + 1, col, 1])
            add_edge(row, col, row + 1, col, target)

    A = coo_matrix(
        (values, (rows, cols)),
        shape=(equation, unknown.sum()),
    ).tocsr()

    solution = lsqr(A, np.asarray(rhs), atol=1e-10, btol=1e-10, iter_lim=3000)[0]
    udf = np.zeros((height, width))
    udf[unknown] = solution
    return np.maximum(udf, 0.0)

udf_no_diffusion = integrate_gradient(no_diffusion_field)
udf_full_vad = integrate_gradient(full_field)
true_udf = np.abs(np.sqrt(xx**2 + yy**2) - 1.0)

error_no_diffusion = np.abs(udf_no_diffusion - true_udf)
error_full_vad = np.abs(udf_full_vad - true_udf)

print(f"No diffusion - mean error: {error_no_diffusion.mean():.6f}")
print(f"Full VAD     - mean error: {error_full_vad.mean():.6f}")

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

for ax, image, title, cmap in [
    (axes[0, 0], udf_no_diffusion, "Ablation: no diffusion", "viridis"),
    (axes[0, 1], udf_full_vad, "Full VAD: diffusion + fusion", "viridis"),
    (axes[1, 0], error_no_diffusion, "Error: no diffusion", "magma"),
    (axes[1, 1], error_full_vad, "Error: full VAD", "magma"),
]:
    colour = ax.contourf(xx, yy, image, levels=60, cmap=cmap)
    ax.scatter(points[:, 0], points[:, 1], s=5, c="white")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(colour, ax=ax)

fig.suptitle("VAD ablation: contribution of diffusion")
fig.tight_layout()

output_path = root / "outputs" / "12_diffusion_ablation.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")