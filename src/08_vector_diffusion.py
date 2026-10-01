"""
Stage 7: VAD vector diffusion.

Create two perturbed copies per input point:
p + epsilon*n with direction +n,
p - epsilon*n with direction -n.
This restores the sign that tensor diffusion cannot store.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
data = np.load(root / "outputs" / "06_optimised_normals.npz")
points = data["points"]
normals = data["vectors"]

grid_size = 256
bound = 1.5
axis = np.linspace(-bound, bound, grid_size)
xx, yy = np.meshgrid(axis, axis)
queries = np.column_stack((xx.ravel(), yy.ravel()))

epsilon = 0.06

# VAD Section 4.4: duplicate each point on both sides of its normal axis.
source_positions = np.concatenate(
    (points + epsilon * normals, points - epsilon * normals), axis=0
)
source_vectors = np.concatenate((normals, -normals), axis=0)

# Build a piecewise vector source field from the nearest perturbed copy.
distances = np.linalg.norm(
    queries[:, None, :] - source_positions[None, :, :], axis=2
)
nearest = distances.argmin(axis=1)
vector_raw = source_vectors[nearest].reshape(grid_size, grid_size, 2)

# Screened-Poisson / heat-style diffusion in Fourier space.
spacing = axis[1] - axis[0]
frequency = 2 * np.pi * np.fft.fftfreq(grid_size, d=spacing)
kx, ky = np.meshgrid(frequency, frequency)
k_squared = kx**2 + ky**2
diffusion_time = 0.015

vector_smooth = np.empty_like(vector_raw)
for component in range(2):
    source_hat = np.fft.fft2(vector_raw[:, :, component])
    vector_smooth[:, :, component] = np.real(
        np.fft.ifft2(source_hat / (1 + diffusion_time * k_squared))
    )

np.savez(
    root / "outputs" / "08_vector_field.npz",
    vector=vector_smooth, xx=xx, yy=yy,
)

def unit_vectors(field):
    magnitude = np.linalg.norm(field, axis=2, keepdims=True)
    return field / np.maximum(magnitude, 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, field, title in [
    (axes[0], vector_raw, "Before vector diffusion"),
    (axes[1], vector_smooth, "After vector diffusion"),
]:
    direction = unit_vectors(field)
    step = 14
    ax.quiver(
        xx[::step, ::step], yy[::step, ::step],
        direction[::step, ::step, 0], direction[::step, ::step, 1],
        color="crimson", angles="xy", scale_units="xy", scale=11,
        width=0.004,
    )
    ax.scatter(points[:, 0], points[:, 1], s=10, c="black", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_title(title)

fig.suptitle("VAD Stage 7: Vector diffusion restores orientation")
fig.tight_layout()

output_path = root / "outputs" / "08_vector_diffusion.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")