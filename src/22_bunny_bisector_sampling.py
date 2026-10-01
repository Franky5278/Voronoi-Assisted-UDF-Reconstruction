from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "bunny_voronoi_10k.npz"

OUTPUT_PATH = ROOT / "data" / "bunny_bisector_samples_200k.npz"

OUTPUT_FIG = ROOT / "outputs" / "22_bunny_bisector_samples.png"

OUTPUT_SUMMARY = ROOT / "outputs" / "22_bunny_bisector_sampling_summary.txt"


# ============================================================
# Settings
# ============================================================

TARGET_TOTAL_SAMPLES = 200_000
RNG_SEED = 42

rng = np.random.default_rng(RNG_SEED)


# ============================================================
# 1. Load Voronoi data
# ============================================================

data = np.load(
    INPUT_PATH,
    allow_pickle=True
)

points = data["points"].astype(np.float64)

vor_vertices = data["vor_vertices"].astype(np.float64)

ridge_points = data["ridge_points"].astype(np.int32)

ridge_vertices = data["ridge_vertices"]

bounded_mask = data["bounded_mask"].astype(bool)

h = float(data["h"][0])


print("Points:", points.shape)
print("Voronoi vertices:", vor_vertices.shape)
print("Ridges:", len(ridge_points))
print("Bounded ridges:", int(bounded_mask.sum()))
print("h =", h)


# ============================================================
# Helper: order polygon vertices in plane
# ============================================================

def order_polygon_vertices(vertices, site_i, site_j):
    """
    Order an unordered set of 3D coplanar polygon vertices.

    The Voronoi bisector plane normal is parallel to:
        p_j - p_i
    """

    center = vertices.mean(axis=0)

    normal = site_j - site_i
    normal /= np.linalg.norm(normal)

    # Choose a vector not parallel to normal
    if abs(normal[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])

    basis_u = np.cross(normal, helper)
    basis_u /= np.linalg.norm(basis_u)

    basis_v = np.cross(normal, basis_u)
    basis_v /= np.linalg.norm(basis_v)

    rel = vertices - center

    x = rel @ basis_u
    y = rel @ basis_v

    angles = np.arctan2(y, x)

    order = np.argsort(angles)

    return vertices[order]


# ============================================================
# Helper: polygon triangulation around centroid
# ============================================================

def polygon_triangles(vertices):
    """
    A convex polygon is triangulated using its centroid.

    Returns:
        center
        triangle vertex pairs
        triangle areas
    """

    center = vertices.mean(axis=0)

    triangles = []
    areas = []

    m = len(vertices)

    for k in range(m):

        a = vertices[k]
        b = vertices[(k + 1) % m]

        area = 0.5 * np.linalg.norm(
            np.cross(a - center, b - center)
        )

        if area > 1e-14:
            triangles.append((a, b))
            areas.append(area)

    return center, triangles, np.asarray(areas)


# ============================================================
# 2. Compute bounded face areas
# ============================================================

face_records = []

print()
print("Computing Voronoi bisector polygon areas...")

for ridge_idx, is_bounded in enumerate(bounded_mask):

    if not is_bounded:
        continue

    i, j = ridge_points[ridge_idx]

    # Safety guard
    if i >= len(points) or j >= len(points):
        continue

    vertex_ids = np.asarray(
        ridge_vertices[ridge_idx],
        dtype=np.int32
    )

    if len(vertex_ids) < 3:
        continue

    vertices = vor_vertices[vertex_ids]

    vertices = order_polygon_vertices(
        vertices,
        points[i],
        points[j]
    )

    center, triangles, tri_areas = polygon_triangles(vertices)

    if len(tri_areas) == 0:
        continue

    face_area = float(tri_areas.sum())

    if not np.isfinite(face_area) or face_area <= 0:
        continue

    face_records.append(
        {
            "ridge_idx": ridge_idx,
            "i": int(i),
            "j": int(j),
            "vertices": vertices,
            "center": center,
            "triangles": triangles,
            "tri_areas": tri_areas,
            "area": face_area,
        }
    )


print("Valid bounded polygon faces:", len(face_records))

if len(face_records) == 0:
    raise RuntimeError("No valid bounded Voronoi faces found.")


face_areas = np.asarray(
    [r["area"] for r in face_records],
    dtype=np.float64
)

total_area = float(face_areas.sum())

print("Total bounded bisector area:", total_area)

print()
print("Face area statistics:")
print("min    =", face_areas.min())
print("median =", np.median(face_areas))
print("mean   =", face_areas.mean())
print("max    =", face_areas.max())


# ============================================================
# 3. Allocate sample counts proportional to face area
#
# Paper:
# samples are uniform over each polygon.
#
# Each sample gets:
#       w(x_k) = polygon_area / samples_on_polygon
# ============================================================

expected_counts = (
    TARGET_TOTAL_SAMPLES
    * face_areas
    / total_area
)

sample_counts = np.floor(expected_counts).astype(np.int64)

# Give at least one sample to each usable polygon.
sample_counts = np.maximum(sample_counts, 1)


# Correct total approximately to target.
difference = TARGET_TOTAL_SAMPLES - int(sample_counts.sum())

if difference > 0:

    fractional = expected_counts - np.floor(expected_counts)

    order = np.argsort(-fractional)

    for idx in order[:difference]:
        sample_counts[idx] += 1


elif difference < 0:

    # Remove samples from largest faces first,
    # but never reduce below 1.
    order = np.argsort(-sample_counts)

    remaining = -difference

    for idx in order:

        removable = sample_counts[idx] - 1

        if removable <= 0:
            continue

        take = min(removable, remaining)

        sample_counts[idx] -= take

        remaining -= take

        if remaining == 0:
            break


actual_total = int(sample_counts.sum())

print()
print("Target sample count:", TARGET_TOTAL_SAMPLES)
print("Actual sample count:", actual_total)


# ============================================================
# 4. Sample points uniformly over polygon areas
#
# Every convex polygon is triangulated.
#
# A triangle is selected proportional to area,
# then sampled uniformly using barycentric coordinates.
# ============================================================

all_samples = np.empty(
    (actual_total, 3),
    dtype=np.float32
)

all_site_i = np.empty(
    actual_total,
    dtype=np.int32
)

all_site_j = np.empty(
    actual_total,
    dtype=np.int32
)

all_weights = np.empty(
    actual_total,
    dtype=np.float32
)

all_face_id = np.empty(
    actual_total,
    dtype=np.int32
)


cursor = 0

print()
print("Sampling bisector polygon surfaces...")


for face_id, record in enumerate(face_records):

    count = int(sample_counts[face_id])

    if count <= 0:
        continue

    tri_areas = record["tri_areas"]

    tri_prob = tri_areas / tri_areas.sum()

    selected_triangles = rng.choice(
        len(tri_prob),
        size=count,
        p=tri_prob
    )

    center = record["center"]

    samples = np.empty(
        (count, 3),
        dtype=np.float64
    )


    for local_idx, tri_idx in enumerate(selected_triangles):

        a, b = record["triangles"][tri_idx]

        # Triangle:
        # center, a, b

        r1 = rng.random()
        r2 = rng.random()

        sqrt_r1 = np.sqrt(r1)

        u = 1.0 - sqrt_r1
        v = sqrt_r1 * (1.0 - r2)
        w = sqrt_r1 * r2

        samples[local_idx] = (
            u * center
            + v * a
            + w * b
        )


    start = cursor
    end = cursor + count

    all_samples[start:end] = samples.astype(np.float32)

    all_site_i[start:end] = record["i"]
    all_site_j[start:end] = record["j"]

    # Paper weight:
    # polygon area / number of samples on polygon
    all_weights[start:end] = (
        record["area"] / count
    )

    all_face_id[start:end] = face_id

    cursor = end


assert cursor == actual_total


# ============================================================
# 5. Save samples
# ============================================================

np.savez_compressed(
    OUTPUT_PATH,

    samples=all_samples,

    site_i=all_site_i,

    site_j=all_site_j,

    weights=all_weights,

    face_id=all_face_id,

    face_area=face_areas.astype(np.float32),

    face_sample_count=sample_counts.astype(np.int32),

    h=np.asarray([h], dtype=np.float64),
)


print()
print("Saved:")
print(OUTPUT_PATH)


# ============================================================
# 6. Sanity checks
# ============================================================

print()
print("Sample shape:", all_samples.shape)

print(
    "Weight range:",
    float(all_weights.min()),
    "to",
    float(all_weights.max())
)

print(
    "Sum of sample weights:",
    float(all_weights.sum())
)

print(
    "Total polygon area:",
    total_area
)

relative_error = abs(
    all_weights.sum() - total_area
) / total_area

print(
    "Area integration relative error:",
    relative_error
)


# ============================================================
# 7. Visualization
#
# Show actual polygonal Voronoi bisectors and sample points
# from a small local group.
# ============================================================

# Find local faces whose two sites are near Bunny top.
seed_idx = int(np.argmax(points[:, 1]))

dist_to_seed = np.linalg.norm(
    points - points[seed_idx],
    axis=1
)

local_sites = set(
    np.argsort(dist_to_seed)[:100]
)

candidate_faces = []

for face_id, record in enumerate(face_records):

    if (
        record["i"] in local_sites
        and record["j"] in local_sites
    ):
        candidate_faces.append(face_id)


candidate_faces = candidate_faces[:10]


fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)


for face_id in candidate_faces:

    record = face_records[face_id]

    verts = record["vertices"]

    # close polygon loop
    loop = np.vstack(
        [verts, verts[0]]
    )

    ax.plot(
        loop[:, 0],
        loop[:, 1],
        loop[:, 2],
        linewidth=1.2
    )

    mask = all_face_id == face_id

    s = all_samples[mask]

    # limit dots for visual clarity
    if len(s) > 200:
        s = s[:200]

    ax.scatter(
        s[:, 0],
        s[:, 1],
        s[:, 2],
        s=8
    )


ax.set_title(
    "Bunny: Samples on Actual 3D Voronoi Bisector Polygons"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_box_aspect((1, 1, 1))

plt.tight_layout()

fig.savefig(
    OUTPUT_FIG,
    dpi=220
)

plt.close(fig)


print()
print("Saved visualization:")
print(OUTPUT_FIG)


# ============================================================
# 8. Summary
# ============================================================

summary = f"""
VAD Bunny 10k - Voronoi Bisector Sampling
==========================================

Input points:
{len(points)}

Valid bounded bisector polygons:
{len(face_records)}

Total bounded bisector area:
{total_area}

Target samples:
{TARGET_TOTAL_SAMPLES}

Actual samples:
{actual_total}

Minimum point spacing h:
{h}

Sum of quadrature weights:
{float(all_weights.sum())}

Relative area integration error:
{relative_error}

Output:
{OUTPUT_PATH}
"""


OUTPUT_SUMMARY.write_text(summary)

print()
print(summary)