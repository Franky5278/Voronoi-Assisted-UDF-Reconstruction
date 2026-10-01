from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "bunny_points_10k.ply"

OUTPUT_NPY = ROOT / "data" / "bunny_points_10k_normalized.npy"
OUTPUT_XYZ = ROOT / "data" / "bunny_points_10k_normalized.xyz"


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_ply_vertices(path):
    """
    Read vertex data from an ASCII or binary little-endian PLY file.
    Only x, y, z are returned.
    """

    with open(path, "rb") as f:

        first = f.readline().decode("ascii").strip()

        if first != "ply":
            raise RuntimeError("This is not a PLY file.")

        fmt = None
        vertex_count = None

        current_element = None
        vertex_properties = []

        while True:
            line = f.readline().decode("ascii").strip()

            if line.startswith("format"):
                fmt = line.split()[1]

            elif line.startswith("element"):
                tokens = line.split()
                current_element = tokens[1]

                if current_element == "vertex":
                    vertex_count = int(tokens[2])

            elif line.startswith("property") and current_element == "vertex":

                tokens = line.split()

                if tokens[1] == "list":
                    raise RuntimeError(
                        "List properties inside vertex element are not supported."
                    )

                data_type = tokens[1]
                name = tokens[2]

                vertex_properties.append((name, data_type))

            elif line == "end_header":
                break

        if vertex_count is None:
            raise RuntimeError("PLY contains no vertex element.")

        print("PLY format:", fmt)
        print("Vertex count from header:", vertex_count)
        print("Vertex properties:", vertex_properties)

        # ----------------------------------------------------
        # ASCII
        # ----------------------------------------------------

        if fmt == "ascii":

            rows = []

            for _ in range(vertex_count):
                values = f.readline().decode("ascii").split()
                rows.append([float(v) for v in values])

            rows = np.asarray(rows)

            names = [name for name, _ in vertex_properties]

            x = rows[:, names.index("x")]
            y = rows[:, names.index("y")]
            z = rows[:, names.index("z")]

            points = np.column_stack((x, y, z))

        # ----------------------------------------------------
        # Binary little endian
        # ----------------------------------------------------

        elif fmt == "binary_little_endian":

            dtype_fields = []

            for name, data_type in vertex_properties:

                if data_type not in PLY_TYPES:
                    raise RuntimeError(
                        f"Unsupported PLY property type: {data_type}"
                    )

                dtype_fields.append(
                    (name, np.dtype(PLY_TYPES[data_type]))
                )

            dtype = np.dtype(dtype_fields)

            data = np.fromfile(
                f,
                dtype=dtype,
                count=vertex_count
            )

            if not all(k in data.dtype.names for k in ["x", "y", "z"]):
                raise RuntimeError("PLY does not contain x/y/z.")

            points = np.column_stack(
                (
                    data["x"],
                    data["y"],
                    data["z"],
                )
            )

        else:

            raise RuntimeError(
                f"Unsupported PLY format: {fmt}"
            )

    return points.astype(np.float64)


# ============================================================
# 1. Load Bunny
# ============================================================

points = load_ply_vertices(INPUT_PATH)

print()
print("Loaded point cloud:", points.shape)

if points.shape != (10000, 3):
    raise RuntimeError(
        f"Expected (10000, 3), but got {points.shape}"
    )


# ============================================================
# 2. Inspect raw bounding box
# ============================================================

pmin = points.min(axis=0)
pmax = points.max(axis=0)

print()
print("Original bounding box:")
print("min =", pmin)
print("max =", pmax)

extent = pmax - pmin

print("extent =", extent)


# ============================================================
# 3. Center the point cloud
# ============================================================

center = (pmin + pmax) / 2.0

points_centered = points - center


# ============================================================
# 4. Uniformly scale longest dimension to 0.8
#
# This means the longest axis fits inside [-0.4, 0.4].
# The same scalar is used for XYZ so geometry is NOT distorted.
# ============================================================

max_extent = extent.max()

scale = 0.8 / max_extent

points_normalized = points_centered * scale


# ============================================================
# 5. Check normalized bounding box
# ============================================================

nmin = points_normalized.min(axis=0)
nmax = points_normalized.max(axis=0)

print()
print("Normalization:")
print("center =", center)
print("scale  =", scale)

print()
print("Normalized bounding box:")
print("min =", nmin)
print("max =", nmax)

print()
print("Number of points:", len(points_normalized))


# ============================================================
# 6. Save
# ============================================================

np.save(
    OUTPUT_NPY,
    points_normalized.astype(np.float32)
)

np.savetxt(
    OUTPUT_XYZ,
    points_normalized,
    fmt="%.8f"
)

print()
print("Saved:")
print(OUTPUT_NPY)
print(OUTPUT_XYZ)