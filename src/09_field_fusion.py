"""
Stage 8: VAD field fusion.

Tensor field: reliable normal axis, but no sign.
Vector field: has sign, but may not be exactly perpendicular.
Fusion: orient the tensor principal axis using the vector field.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]

tensor_data = np.load(root / "outputs" / "07_tensor_field.npz")
vector_data = np.load(root / "outputs" / "08_vector_field.npz")

tensor = tensor_data["tensor"]
vector = vector_data["vector"]
xx = tensor_data["xx"]
yy = tensor_data["yy"]

# Principal axis of tensor field.
_, eigenvectors = np.linalg.eigh(tensor)
principal_axis = eigenvectors[:, :, :, 1]

# Give each axis a sign using the diffused vector field.
alignment = np.sum(principal_axis * vector, axis=2, keepdims=True)
fused_field = np.where(alignment >= 0, principal_axis, -principal_axis)

np.savez(
    root / "outputs" / "09_fused_field.npz",
    field=fused_field, xx=xx, yy=yy,
)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
step = 14

# Tensor principal axis: blue lines, no arrowhead.
ax = axes[0]
for row in range(6, xx.shape[0], step):
    for col in range(6, xx.shape[1], step):
        x, y = xx[row, col], yy[row, col]
        dx, dy = principal_axis[row, col] * 0.07
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color="royalblue", lw=1)
ax.set_title("Tensor principal axes: sign ambiguous")
ax.set_aspect("equal")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Fused field: red arrows, now sign-consistent.
ax = axes[1]
ax.quiver(
    xx[::step, ::step], yy[::step, ::step],
    fused_field[::step, ::step, 0],
    fused_field[::step, ::step, 1],
    color="crimson", angles="xy", scale_units="xy", scale=11,
    width=0.004,
)
ax.set_title(r"Fused field $Y_f \approx \nabla u$")
ax.set_aspect("equal")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

fig.suptitle("VAD Stage 8: Tensor-vector field fusion")
fig.tight_layout()

output_path = root / "outputs" / "09_field_fusion.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")