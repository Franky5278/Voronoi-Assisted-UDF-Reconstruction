from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

root = Path(__file__).resolve().parents[1]

field_data = np.load(root / "outputs" / "09_fused_field.npz")
normal_data = np.load(root / "outputs" / "06_optimised_normals.npz")

field = field_data["field"]
xx, yy = field_data["xx"], field_data["yy"]
points = normal_data["points"]

height, width = xx.shape
spacing = xx[0, 1] - xx[0, 0]

# Grid cells closest to input samples enforce u(p_i)=0.
constraint = np.zeros((height, width), dtype=bool)
for px, py in points:
    col = int(np.clip(round((px - xx[0, 0]) / spacing), 0, width - 1))
    row = int(np.clip(round((py - yy[0, 0]) / spacing), 0, height - 1))
    constraint[row, col] = True

unknown = ~constraint
index = -np.ones((height, width), dtype=int)
index[unknown] = np.arange(unknown.sum())

rows, cols, values, rhs = [], [], [], []
equation = 0

def add_gradient_equation(a_row, a_col, b_row, b_col, target):
    """Enforce u(b) - u(a) approximately equals target."""
    global equation

    a_unknown = unknown[a_row, a_col]
    b_unknown = unknown[b_row, b_col]

    if not a_unknown and not b_unknown:
        return

    if a_unknown:
        rows.append(equation)
        cols.append(index[a_row, a_col])
        values.append(-1.0)

    if b_unknown:
        rows.append(equation)
        cols.append(index[b_row, b_col])
        values.append(1.0)

    rhs.append(target)
    equation += 1

# Horizontal gradient equations: du/dx = Y_x.
for row in range(height):
    for col in range(width - 1):
        target = spacing * 0.5 * (field[row, col, 0] + field[row, col + 1, 0])
        add_gradient_equation(row, col, row, col + 1, target)

# Vertical gradient equations: du/dy = Y_y.
for row in range(height - 1):
    for col in range(width):
        target = spacing * 0.5 * (field[row, col, 1] + field[row + 1, col, 1])
        add_gradient_equation(row, col, row + 1, col, target)

A = coo_matrix(
    (values, (rows, cols)),
    shape=(equation, unknown.sum()),
).tocsr()

solution = lsqr(A, np.asarray(rhs), atol=1e-10, btol=1e-10, iter_lim=3000)[0]

udf_raw = np.zeros((height, width))
udf_raw[unknown] = solution
udf = np.maximum(udf_raw, 0.0)

true_udf = np.abs(np.sqrt(xx**2 + yy**2) - 1.0)
error = np.abs(udf - true_udf)

print(f"Mean absolute UDF error: {error.mean():.6f}")
print(f"Maximum UDF error:      {error.max():.6f}")
print(f"Negative values before clipping: {(udf_raw < 0).mean():.2%}")

np.savez(root / "outputs" / "10_reconstructed_udf.npz", udf=udf, xx=xx, yy=yy)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

for ax, image, title, cmap in [
    (axes[0], udf, "Reconstructed UDF from fused field", "viridis"),
    (axes[1], true_udf, "True UDF of the unit circle", "viridis"),
    (axes[2], error, "Absolute reconstruction error", "magma"),
]:
    colour = ax.contourf(xx, yy, image, levels=60, cmap=cmap)
    ax.contour(xx, yy, image, levels=[0.02], colors="white", linewidths=1.2)
    ax.scatter(points[:, 0], points[:, 1], s=5, c="white")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(colour, ax=ax)

fig.suptitle("VAD Stage 9: UDF recovery by gradient integration")
fig.tight_layout()

output_path = root / "outputs" / "10_poisson_udf.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")