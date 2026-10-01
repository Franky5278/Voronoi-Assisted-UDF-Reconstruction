from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, cKDTree


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "bunny_points_10k_normalized.npy"

OUTPUT_NPZ = ROOT / "data" / "bunny_voronoi_10k.npz"

OUTPUT_TXT = ROOT / "outputs" / "21_bunny_voronoi_summary.txt"

OUTPUT_FIG = ROOT / "outputs" / "21_bunny_voronoi_local_connectivity.png"


# ============================================================
# 1. Load normalized Bunny point cloud
# ============================================================

points = np.load(INPUT_PATH).astype(np.float64)

print("Loaded:", points.shape)

if points.shape != (10000, 3):
    raise RuntimeError(
        f"Expected (10000, 3), got {points.shape}"
    )

n = len(points)


# ============================================================
# 2. Compute nearest-neighbour scale h
#
# Paper:
# h = minimum distance between any two input points
#
# This will later be used for:
#
# t = h^2
# epsilon = 1e-4 * h
# ============================================================

print()
print("Computing nearest-neighbour distances...")

tree = cKDTree(points)

distances, indices = tree.query(points, k=2)

# [:, 0] is each point itself, distance = 0
nearest = distances[:, 1]

h = float(nearest.min())

print("Minimum nearest-neighbour distance h =", h)
print("Median nearest-neighbour distance    =", float(np.median(nearest)))
print("Mean nearest-neighbour distance      =", float(np.mean(nearest)))

if h <= 0:
    raise RuntimeError(
        "h <= 0. Duplicate points may exist."
    )


# ============================================================
# 3. Build 3D Voronoi diagram
#
# QJ:
# slightly perturbs numerically degenerate configurations
# to make QHull more robust for surface-sampled point clouds.
# ============================================================

print()
print("Constructing 3D Voronoi diagram...")

vor = Voronoi(
    points,
    qhull_options="Qbb Qc Qz QJ"
)

print("3D Voronoi construction finished.")


# ============================================================
# 4. Extract Voronoi adjacency
#
# ridge_points[k] = [i, j]
#
# means:
# Voronoi cell of point i and point j share a bisector.
#
# In 3D, the bisector is generally a polygonal face.
# ============================================================

ridge_points = np.asarray(
    vor.ridge_points,
    dtype=np.int32
)

ridge_vertices_raw = vor.ridge_vertices

num_ridges = len(ridge_points)

print()
print("Number of Voronoi bisectors / ridges:", num_ridges)


# ============================================================
# 5. Determine bounded Voronoi bisectors
#
# A ridge containing -1 extends to infinity.
#
# For our optimization later, finite polygonal faces are
# particularly convenient for direct surface sampling.
# ============================================================

bounded_mask = []

bounded_count = 0
unbounded_count = 0

polygon_sizes = []

for rv in ridge_vertices_raw:

    rv = np.asarray(rv, dtype=np.int32)

    bounded = (
        len(rv) >= 3
        and np.all(rv >= 0)
    )

    bounded_mask.append(bounded)

    if bounded:
        bounded_count += 1
        polygon_sizes.append(len(rv))
    else:
        unbounded_count += 1

bounded_mask = np.asarray(
    bounded_mask,
    dtype=bool
)


print("Bounded bisectors:  ", bounded_count)
print("Unbounded bisectors:", unbounded_count)


if polygon_sizes:

    polygon_sizes = np.asarray(polygon_sizes)

    print()
    print("Bounded polygon vertex counts:")
    print("min    =", polygon_sizes.min())
    print("median =", np.median(polygon_sizes))
    print("mean   =", polygon_sizes.mean())
    print("max    =", polygon_sizes.max())


# ============================================================
# 6. Compute number of Voronoi neighbours per point
# ============================================================

degree = np.zeros(n, dtype=np.int32)

for i, j in ridge_points:

    # Guard against potential auxiliary indices
    if i < n and j < n:

        degree[i] += 1
        degree[j] += 1


print()
print("Voronoi neighbour statistics:")
print("min    =", degree.min())
print("median =", np.median(degree))
print("mean   =", degree.mean())
print("max    =", degree.max())


# ============================================================
# 7. Save Voronoi structure
#
# ridge_vertices has variable length, therefore store
# it as an object array.
# ============================================================

ridge_vertices = np.asarray(
    [
        np.asarray(rv, dtype=np.int32)
        for rv in ridge_vertices_raw
    ],
    dtype=object
)


np.savez_compressed(
    OUTPUT_NPZ,

    points=points.astype(np.float32),

    vor_vertices=vor.vertices.astype(np.float32),

    ridge_points=ridge_points,

    ridge_vertices=ridge_vertices,

    bounded_mask=bounded_mask,

    degree=degree,

    nearest_distance=nearest.astype(np.float32),

    h=np.asarray([h], dtype=np.float64),
)


print()
print("Saved Voronoi data:")
print(OUTPUT_NPZ)


# ============================================================
# 8. Create a LOCAL connectivity visualization
#
# Full 10k Voronoi diagram is visually overwhelming.
#
# We select a local patch near the point with maximum Y
# (roughly upper part of Bunny), and plot Voronoi-site
# connectivity.
#
# IMPORTANT:
# These lines are site adjacency induced by Voronoi cells,
# NOT the actual Voronoi polygonal bisector surfaces.
# ============================================================

seed_idx = int(np.argmax(points[:, 1]))

patch_size = 180

_, patch_idx = tree.query(
    points[seed_idx],
    k=patch_size
)

patch_set = set(
    int(x)
    for x in np.atleast_1d(patch_idx)
)


fig = plt.figure(figsize=(9, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)


# Plot local points
patch_points = points[
    np.asarray(list(patch_set))
]

ax.scatter(
    patch_points[:, 0],
    patch_points[:, 1],
    patch_points[:, 2],
    s=12
)


# Plot Voronoi-site adjacency
edge_count = 0

for i, j in ridge_points:

    if i in patch_set and j in patch_set:

        p0 = points[i]
        p1 = points[j]

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            linewidth=0.6
        )

        edge_count += 1


ax.set_title(
    "Bunny 10k: Local Voronoi-site Connectivity"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_box_aspect((1, 1, 1))

plt.tight_layout()

fig.savefig(
    OUTPUT_FIG,
    dpi=200
)

plt.close(fig)


print()
print("Local connectivity edges drawn:", edge_count)

print("Saved visualization:")
print(OUTPUT_FIG)


# ============================================================
# 9. Write summary
# ============================================================

summary = f"""
VAD Bunny 10k - 3D Voronoi Summary
====================================

Input points:
{n}

Number of Voronoi vertices:
{len(vor.vertices)}

Number of Voronoi bisectors:
{num_ridges}

Bounded bisectors:
{bounded_count}

Unbounded bisectors:
{unbounded_count}

Voronoi neighbours per point:
min     = {degree.min()}
median  = {np.median(degree)}
mean    = {degree.mean()}
max     = {degree.max()}

Nearest-neighbour scale:
h       = {h}

Future VAD diffusion parameters:
t       = h^2       = {h ** 2}
epsilon = 1e-4 * h  = {1e-4 * h}
"""


OUTPUT_TXT.write_text(summary)

print()
print(summary)

print("Saved summary:")
print(OUTPUT_TXT)