"""
Stage 6: tensor diffusion, corresponding to VAD Section 4.4.

n and -n are equivalent, so we encode each normal axis as:
T = n outer-product n.
Then solve a screened-Poisson / heat-style diffusion equation in Fourier space.
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

# Each grid position inherits its nearest point's rank-one tensor n outer-product n.
distances = np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=2)
nearest = distances.argmin(axis=1)
tensor_raw = np.einsum("ni,nj->nij", normals[nearest], normals[nearest])
tensor_raw = tensor_raw.reshape(grid_size, grid_size, 2, 2)

# Paper Eq. (12): (Delta - 1/t) Y_t = -(1/t) T.
# Fourier solution: Y_hat = T_hat / (1 + t * |k|^2).
spacing = axis[1] - axis[0]
frequencies = 2 * np.pi * np.fft.fftfreq(grid_size, d=spacing)
kx, ky = np.meshgrid(frequencies, frequencies)
k_squared = kx**2 + ky**2
diffusion_time = 0.015

tensor_smooth = np.empty_like(tensor_raw)
for a in range(2):
    for b in range(2):
        source_hat = np.fft.fft2(tensor_raw[:, :, a, b])
        tensor_smooth[:, :, a, b] = np.real(
            np.fft.ifft2(source_hat / (1 + diffusion_time * k_squared))
        )

# Principal eigenvector = the diffused bidirectional normal AXIS.
eigenvalues, eigenvectors = np.linalg.eigh(tensor_smooth)
principal_axis = eigenvectors[:, :, :, 1]
anisotropy = eigenvalues[:, :, 1] - eigenvalues[:, :, 0]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, field, title in [
    (axes[0], tensor_raw, "Before diffusion: piecewise Voronoi tensor field"),
    (axes[1], tensor_smooth, "After tensor diffusion: smooth normal axes"),
]:
    _, vectors = np.linalg.eigh(field)
    directions = vectors[:, :, :, 1]

    ax.scatter(points[:, 0], points[:, 1], s=10, c="black", zorder=3)
    for row in range(6, grid_size, 14):
        for col in range(6, grid_size, 14):
            x, y = xx[row, col], yy[row, col]
            dx, dy = directions[row, col] * 0.07
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color="royalblue", lw=1)

    ax.set_aspect("equal")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_title(title)

fig.suptitle(r"VAD Stage 6: Tensor diffusion of $T=n\otimes n$")
fig.tight_layout()

output_path = root / "outputs" / "07_tensor_diffusion.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
np.savez(root / "outputs" / "07_tensor_field.npz", tensor=tensor_smooth, xx=xx, yy=yy)
print(f"Saved figure to: {output_path}")
print(f"Mean tensor anisotropy: {anisotropy.mean():.6f}")

