"""
Stage 5: 2D VAD-style bidirectional-normal optimisation.

Uses:
- sampled Voronoi-bisector consistency terms E_d and E_g;
- Adam, as in VAD;
- a simple 2D local-orthogonality regulariser for stability.

This is a controlled 2D reproduction module, not yet the paper's full 3D solver.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(7)
device = "cuda" if torch.cuda.is_available() else "cpu"

num_points = 40
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
points_np = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
points = torch.tensor(points_np, device=device)

# Voronoi-neighbour pairs for the uniformly sampled circle.
i = torch.arange(num_points, device=device)
j = (i + 1) % num_points

# Sample every neighbour-pair bisector from centre towards its exterior.
ray = points[i] + points[j]
ray = ray / torch.linalg.norm(ray, dim=1, keepdim=True)
t = torch.linspace(0.02, 1.5, 80, device=device)
samples = ray[:, None, :] * t[None, :, None]

# Local chord directions; normals should be perpendicular to these chords.
chord = points[j] - points[i]
chord = chord / torch.linalg.norm(chord, dim=1, keepdim=True)

# Each angle theta_i represents the bidirectional pair {n_i, -n_i}.
theta = torch.rand(num_points, device=device, requires_grad=True) * (2 * np.pi)
theta = torch.nn.Parameter(theta)

optimizer = torch.optim.Adam([theta], lr=0.03)
history = []

def vectors_from_theta():
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)

def energy(vectors):
    vi = vectors[i]
    vj = vectors[j]

    dot_i = ((samples - points[i, None, :]) * vi[:, None, :]).sum(dim=2)
    dot_j = ((samples - points[j, None, :]) * vj[:, None, :]).sum(dim=2)

    # Paper-like E_d: field values should agree on B_ij.
    ed = torch.abs(torch.abs(dot_i) - torch.abs(dot_j)).mean()

    # Paper-like E_g: gradients should agree on B_ij.
    grad_i = torch.sign(dot_i)[..., None] * vi[:, None, :]
    grad_j = torch.sign(dot_j)[..., None] * vj[:, None, :]
    eg = torch.linalg.norm(grad_i - grad_j, dim=2).mean()

    # 2D toy stabiliser: normal should be perpendicular to local chord.
    ea = ((vi * chord).sum(dim=1) ** 2).mean()

    return ed, eg, ea

with torch.no_grad():
    initial_vectors = vectors_from_theta().cpu().numpy()
    initial_terms = [x.item() for x in energy(vectors_from_theta())]

for step in range(1000):
    optimizer.zero_grad()
    ed, eg, ea = energy(vectors_from_theta())
    total = ed + 0.1 * eg + 5.0 * ea
    total.backward()
    optimizer.step()
    history.append(total.item())

with torch.no_grad():
    final_vectors = vectors_from_theta().cpu().numpy()
    final_terms = [x.item() for x in energy(vectors_from_theta())]

print("Initial  E_d, E_g, E_align:", initial_terms)
print("Final    E_d, E_g, E_align:", final_terms)

show = np.arange(0, num_points, 2)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

for ax, vectors, title in [
    (axes[0], initial_vectors, "Random initial vectors"),
    (axes[1], final_vectors, "After Adam optimisation"),
]:
    ax.scatter(points_np[:, 0], points_np[:, 1], s=12, c="black")
    ax.quiver(
        points_np[show, 0], points_np[show, 1],
        vectors[show, 0], vectors[show, 1],
        color="crimson", scale=12, width=0.006,
    )
    ax.quiver(
        points_np[show, 0], points_np[show, 1],
        -vectors[show, 0], -vectors[show, 1],
        color="royalblue", scale=12, width=0.006,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(alpha=0.25)

axes[2].plot(history, color="darkgreen")
axes[2].set_title("Optimisation loss")
axes[2].set_xlabel("Adam iteration")
axes[2].set_ylabel(r"$E_d + 0.1E_g + 5E_{align}$")
axes[2].grid(alpha=0.25)

fig.suptitle("VAD Stage 5: Optimising bidirectional normals")
fig.tight_layout()

output_path = Path(__file__).resolve().parents[1] / "outputs" / "06_normal_optimisation.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

normal_path = Path(__file__).resolve().parents[1] / "outputs" / "06_optimised_normals.npz"
np.savez(normal_path, points=points_np, vectors=final_vectors)
print(f"Saved optimised normals to: {normal_path}")