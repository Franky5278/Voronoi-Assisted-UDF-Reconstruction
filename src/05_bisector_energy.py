"""
Stage 4: evaluate VAD's bisector consistency energy in 2D.

E_d: field-value difference across a Voronoi bisector.
E_g: gradient difference across the same bisector.

This is a sampled 2D analogue of paper Eqs. (2), (3), (5), and (6).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

num_points = 40
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
points = np.column_stack((np.cos(angles), np.sin(angles)))

# Controlled reference vectors for the circle.
true_vectors = points.copy()

# VAD-style random initialization.
rng = np.random.default_rng(7)
random_angles = rng.uniform(0, 2 * np.pi, num_points)
random_vectors = np.column_stack((np.cos(random_angles), np.sin(random_angles)))


def field_and_gradient(x, point, vector):
    """
    Paper Eq. (1): F_i(x) = |(x-p_i) dot v_i| for unit v_i.
    The gradient is sign((x-p_i) dot v_i) * v_i.
    """
    dot = np.sum((x - point) * vector, axis=-1)
    value = np.abs(dot)
    gradient = np.sign(dot)[..., None] * vector
    return value, gradient


def bisector_energy(vectors):
    value_errors = []
    gradient_errors = []

    # On the circle, Voronoi neighbours are adjacent point indices.
    for i in range(num_points):
        j = (i + 1) % num_points

        # This ray starts at the circle centre and lies on B_ij.
        ray_direction = points[i] + points[j]
        ray_direction /= np.linalg.norm(ray_direction)
        samples = np.linspace(0.02, 1.5, 80)[:, None] * ray_direction

        fi, gi = field_and_gradient(samples, points[i], vectors[i])
        fj, gj = field_and_gradient(samples, points[j], vectors[j])

        value_errors.append(np.abs(fi - fj))
        gradient_errors.append(np.linalg.norm(gi - gj, axis=1))

    return np.concatenate(value_errors), np.concatenate(gradient_errors)


for name, vectors in [
    ("Correct normals", true_vectors),
    ("Random vectors", random_vectors),
]:
    ed, eg = bisector_energy(vectors)
    print(f"{name:16s}  E_d ≈ {ed.mean():.6f}   E_g ≈ {eg.mean():.6f}")


# Visualise one representative Voronoi bisector.
i, j = 0, 1
ray = points[i] + points[j]
ray /= np.linalg.norm(ray)
samples = np.linspace(0.02, 1.5, 250)[:, None] * ray
distance_from_centre = np.linalg.norm(samples, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, vectors, title in [
    (axes[0], true_vectors, "Correct bidirectional normals"),
    (axes[1], random_vectors, "Random initial vectors"),
]:
    fi, _ = field_and_gradient(samples, points[i], vectors[i])
    fj, _ = field_and_gradient(samples, points[j], vectors[j])

    ax.plot(distance_from_centre, fi, label=r"$F^i_{B_{ij}}(x)$", lw=2)
    ax.plot(distance_from_centre, fj, "--", label=r"$F^j_{B_{ij}}(x)$", lw=2)
    ax.set_title(title)
    ax.set_xlabel("position along one Voronoi bisector")
    ax.set_ylabel("projection distance")
    ax.grid(alpha=0.25)
    ax.legend()

fig.suptitle("VAD Stage 4: field consistency on a Voronoi bisector")
fig.tight_layout()

output_path = Path(__file__).resolve().parents[1] / "outputs" / "05_bisector_energy.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved figure to: {output_path}")