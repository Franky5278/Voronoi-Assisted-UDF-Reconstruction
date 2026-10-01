import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Voronoi

root = Path(__file__).resolve().parents[1]
output_path = root / "outputs" / "19_noisy_position_optimisation.png"

source = np.load(root / "outputs" / "18_noisy_ellipse_input.npz")
clean_points = source["clean_points"]
noisy_points = source["noisy_points"]

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(30)
rng = np.random.default_rng(30)

domain_radius = 2.0
lambda_g = 1e-3
lambda_p = 1e2
outer_iterations = 3

def build_bisector_samples(points_np):
    """为当前点位置重算 Voronoi，并仅保留真实 B_ij 上的采样点。"""
    vor = Voronoi(points_np)
    pairs = np.unique(np.sort(vor.ridge_points, axis=1), axis=0)

    sample_list = []
    pair_list = []

    for i, j in pairs:
        pi, pj = points_np[i], points_np[j]
        midpoint = 0.5 * (pi + pj)

        edge = pj - pi
        edge /= np.linalg.norm(edge)
        direction = np.array([-edge[1], edge[0]])

        candidates = (
            midpoint
            + np.linspace(-domain_radius, domain_radius, 241)[:, None] * direction
        )

        distances = np.linalg.norm(
            candidates[:, None, :] - points_np[None, :, :],
            axis=2,
        )
        nearest_two = np.argpartition(distances, kth=1, axis=1)[:, :2]

        valid = np.array([
            set(two) == {i, j}
            for two in nearest_two
        ])
        samples = candidates[valid]

        if len(samples) >= 5:
            chosen = samples[::max(1, len(samples) // 18)]
            sample_list.append(chosen)
            pair_list.extend([(i, j)] * len(chosen))

    return np.vstack(sample_list), np.asarray(pair_list, dtype=np.int64)

def mean_nearest_neighbour_distance(points_np):
    distance = np.linalg.norm(
        points_np[:, None, :] - points_np[None, :, :],
        axis=2,
    )
    distance[distance == 0] = np.inf
    return distance.min(axis=1).mean()

def optimise_normals(points_np, iterations=600):
    """
    论文 Eq.(5)(6) 的 2D 离散实现：
    随机双向轴 -> Adam 最小化 E_d + lambda_g E_g。
    """
    samples_np, pair_ids_np = build_bisector_samples(points_np)
    h = mean_nearest_neighbour_distance(points_np)
    weight = 2 * domain_radius / 241

    points = torch.tensor(points_np, dtype=torch.float32, device=device)
    samples = torch.tensor(samples_np, dtype=torch.float32, device=device)
    pair_ids = torch.tensor(pair_ids_np, dtype=torch.long, device=device)

    raw = torch.nn.Parameter(
        torch.randn(len(points_np), 2, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([raw], lr=0.04)

    history = []

    for _ in range(iterations):
        optimizer.zero_grad()

        normals = raw / (
            torch.linalg.norm(raw, dim=1, keepdim=True) + 1e-12
        )

        i = pair_ids[:, 0]
        j = pair_ids[:, 1]

        vi, vj = normals[i], normals[j]
        ri = samples - points[i]
        rj = samples - points[j]

        signed_i = torch.sum(ri * vi, dim=1)
        signed_j = torch.sum(rj * vj, dim=1)

        fi = torch.abs(signed_i)
        fj = torch.abs(signed_j)

        grad_i = torch.sign(signed_i).unsqueeze(1) * vi
        grad_j = torch.sign(signed_j).unsqueeze(1) * vj

        E_d = (weight / h**2) * torch.sum(torch.abs(fi - fj))
        E_g = weight * torch.sum(
            torch.linalg.norm(grad_i - grad_j, dim=1)
        )

        loss = E_d + lambda_g * E_g
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    normals = (
        raw.detach()
        / (torch.linalg.norm(raw.detach(), dim=1, keepdim=True) + 1e-12)
    ).cpu().numpy()

    return normals, history

def optimise_offsets(current_points, original_points, normals, iterations=350):
    """
    论文 Eq.(9)(10) 的位置步：
    法向固定，只允许 p_i 沿 n_i 移动；
    E_offset = E_d + lambda_p * E_reg。
    """
    samples_np, pair_ids_np = build_bisector_samples(current_points)
    h = mean_nearest_neighbour_distance(current_points)
    weight = 2 * domain_radius / 241

    current = torch.tensor(
        current_points, dtype=torch.float32, device=device
    )
    original = torch.tensor(
        original_points, dtype=torch.float32, device=device
    )
    normals_t = torch.tensor(
        normals, dtype=torch.float32, device=device
    )
    samples = torch.tensor(
        samples_np, dtype=torch.float32, device=device
    )
    pair_ids = torch.tensor(
        pair_ids_np, dtype=torch.long, device=device
    )

    offsets = torch.nn.Parameter(
        torch.zeros(len(current_points), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([offsets], lr=0.004)

    history = []

    for _ in range(iterations):
        optimizer.zero_grad()

        updated = current + offsets.unsqueeze(1) * normals_t

        i = pair_ids[:, 0]
        j = pair_ids[:, 1]

        fi = torch.abs(
            torch.sum((samples - updated[i]) * normals_t[i], dim=1)
        )
        fj = torch.abs(
            torch.sum((samples - updated[j]) * normals_t[j], dim=1)
        )

        E_d = (weight / h**2) * torch.sum(torch.abs(fi - fj))

        # 论文 Eq.(10)：全局锚点 + 局部偏移阻尼
        E_reg = (
            torch.mean(torch.sum((updated - original) ** 2, dim=1))
            + torch.mean(offsets ** 2)
        )

        loss = E_d + lambda_p * E_reg
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    updated_points = (
        current + offsets.unsqueeze(1) * normals_t
    ).detach().cpu().numpy()

    return updated_points, offsets.detach().cpu().numpy(), history

# ------------------------------------------------------------
# 外循环：normal -> position -> recompute Voronoi
# ------------------------------------------------------------
current_points = noisy_points.copy()
normal_histories = []
offset_histories = []
all_offsets = []

for outer in range(outer_iterations):
    normals, normal_history = optimise_normals(current_points)
    current_points, offsets, offset_history = optimise_offsets(
        current_points=current_points,
        original_points=noisy_points,
        normals=normals,
    )

    normal_histories.append(normal_history)
    offset_histories.append(offset_history)
    all_offsets.append(offsets)

    print(
        f"Outer iteration {outer + 1}/{outer_iterations}: "
        f"mean |offset|={np.mean(np.abs(offsets)):.6f}"
    )

# 最后一次位置更新后，按论文流程重新优化一次法向
final_normals, final_normal_history = optimise_normals(current_points)
normal_histories.append(final_normal_history)

noisy_rmse = np.sqrt(np.mean(np.sum(
    (noisy_points - clean_points) ** 2, axis=1
)))
rectified_rmse = np.sqrt(np.mean(np.sum(
    (current_points - clean_points) ** 2, axis=1
)))

# ------------------------------------------------------------
# 可视化
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

axes[0].scatter(
    clean_points[:, 0], clean_points[:, 1],
    c="tab:green", s=20, label="clean reference",
)
axes[0].scatter(
    noisy_points[:, 0], noisy_points[:, 1],
    c="crimson", s=18, label="noisy input",
)
axes[0].set_title("Before rectification")
axes[0].set_aspect("equal")
axes[0].set_xlim(-1.7, 1.7)
axes[0].set_ylim(-1.7, 1.7)
axes[0].legend()

axes[1].scatter(
    clean_points[:, 0], clean_points[:, 1],
    c="tab:green", s=20, label="clean reference",
)
axes[1].scatter(
    current_points[:, 0], current_points[:, 1],
    c="royalblue", s=18, label="rectified points",
)
for old, new in zip(noisy_points, current_points):
    axes[1].plot(
        [old[0], new[0]],
        [old[1], new[1]],
        color="gray", alpha=0.35, lw=0.7,
    )
axes[1].set_title("After normal-guided rectification")
axes[1].set_aspect("equal")
axes[1].set_xlim(-1.7, 1.7)
axes[1].set_ylim(-1.7, 1.7)
axes[1].legend()

for point, normal in zip(current_points, final_normals):
    axes[2].scatter(point[0], point[1], c="black", s=10)
    axes[2].arrow(
        point[0], point[1],
        0.10 * normal[0], 0.10 * normal[1],
        color="crimson", width=0.002,
        head_width=0.035, length_includes_head=True,
    )
    axes[2].arrow(
        point[0], point[1],
        -0.10 * normal[0], -0.10 * normal[1],
        color="royalblue", width=0.002,
        head_width=0.035, length_includes_head=True,
    )
axes[2].set_title("Final bidirectional normals")
axes[2].set_aspect("equal")
axes[2].set_xlim(-1.7, 1.7)
axes[2].set_ylim(-1.7, 1.7)

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.suptitle(
    "Paper-style alternating normal and position optimisation",
    y=1.02,
)
fig.tight_layout()
fig.savefig(output_path, dpi=180, bbox_inches="tight")

np.savez(
    root / "outputs" / "19_rectified_noisy_ellipse.npz",
    clean_points=clean_points,
    noisy_points=noisy_points,
    rectified_points=current_points,
    normals=final_normals,
    noisy_rmse=noisy_rmse,
    rectified_rmse=rectified_rmse,
)

print(f"Noisy point RMSE:      {noisy_rmse:.6f}")
print(f"Rectified point RMSE:  {rectified_rmse:.6f}")
print(f"Saved figure to: {output_path}")