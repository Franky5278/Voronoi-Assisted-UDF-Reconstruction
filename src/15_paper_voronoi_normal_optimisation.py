import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Voronoi

root = Path(__file__).resolve().parents[1]
output_path = root / "outputs" / "15_paper_voronoi_normal_optimisation.png"

# ------------------------------------------------------------
# 输入：无向椭圆点云；论文会从随机双向向量开始
# ------------------------------------------------------------
rng = np.random.default_rng(12)
torch.manual_seed(12)

a, b = 1.20, 0.72
n_points = 80
theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
points_np = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
h = np.linalg.norm(points_np - np.roll(points_np, -1, axis=0), axis=1).mean()

# ------------------------------------------------------------
# 严格对应论文：构建 Voronoi diagram，沿实际 bisector 采样
# 2D 中 bisector 是线段/射线；这里保留“这两个点确实最近”的采样点
# ------------------------------------------------------------
vor = Voronoi(points_np)
pairs = np.unique(np.sort(vor.ridge_points, axis=1), axis=0)

all_samples = []
all_pair_ids = []
domain_radius = 2.0

for i, j in pairs:
    pi, pj = points_np[i], points_np[j]
    midpoint = 0.5 * (pi + pj)

    # 过 midpoint 且垂直于 p_i-p_j 的直线
    direction = pj - pi
    direction /= np.linalg.norm(direction)
    bisector_direction = np.array([-direction[1], direction[0]])

    candidates = midpoint + np.linspace(-domain_radius, domain_radius, 241)[:, None] * bisector_direction

    # 只保留实际位于 B_ij 的位置：
    # i、j 必须恰好是离该位置最近的两个点
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

    # 每条 bisector 至少取少量均匀样本
    if len(samples) >= 5:
        chosen = samples[::max(1, len(samples) // 18)]
        all_samples.append(chosen)
        all_pair_ids.extend([(i, j)] * len(chosen))

samples_np = np.vstack(all_samples)
pair_ids_np = np.asarray(all_pair_ids, dtype=np.int64)

# 对 2D bisector 的离散积分权重：每条线段长度 / 该线段采样数
# 不同 bisector 的采样数近似相同，因此使用统一局部长度权重
weight = 2 * domain_radius / 241

print(f"Voronoi neighbour pairs: {len(pairs)}")
print(f"Valid bisector samples: {len(samples_np)}")
print(f"Mean nearest-neighbour distance h: {h:.6f}")

# ------------------------------------------------------------
# 论文 Eq.(1), Eq.(5), Eq.(6)
# d_{p,v}(x)= |(x-p)·v| / ||v||
# ∇d = sign((x-p)·v) * v / ||v||
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
points = torch.tensor(points_np, dtype=torch.float32, device=device)
samples = torch.tensor(samples_np, dtype=torch.float32, device=device)
pair_ids = torch.tensor(pair_ids_np, dtype=torch.long, device=device)

raw_vectors = torch.nn.Parameter(
    torch.randn(n_points, 2, dtype=torch.float32, device=device)
)
initial_vectors = raw_vectors.detach().cpu().numpy().copy()

optimizer = torch.optim.Adam([raw_vectors], lr=0.04)
losses, ed_history, eg_history = [], [], []

lambda_d = 1.0
lambda_g = 1e-3  # 论文 clean-input 设置

for iteration in range(1000):
    optimizer.zero_grad()

    vectors = raw_vectors / (torch.linalg.norm(raw_vectors, dim=1, keepdim=True) + 1e-12)

    i = pair_ids[:, 0]
    j = pair_ids[:, 1]

    vi, vj = vectors[i], vectors[j]
    ri = samples - points[i]
    rj = samples - points[j]

    signed_i = torch.sum(ri * vi, dim=1)
    signed_j = torch.sum(rj * vj, dim=1)

    # Eq.(1)
    fi = torch.abs(signed_i)
    fj = torch.abs(signed_j)

    # 对 x 的梯度；在 0 附近设 sign(0)=0
    grad_i = torch.sign(signed_i).unsqueeze(1) * vi
    grad_j = torch.sign(signed_j).unsqueeze(1) * vj

    # Eq.(5), Eq.(6) 的 2D 离散版本
    E_d = (weight / h**2) * torch.sum(torch.abs(fi - fj))
    E_g = weight * torch.sum(torch.linalg.norm(grad_i - grad_j, dim=1))

    loss = lambda_d * E_d + lambda_g * E_g
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    ed_history.append(E_d.item())
    eg_history.append(E_g.item())

final_vectors = (
    raw_vectors.detach()
    / (torch.linalg.norm(raw_vectors.detach(), dim=1, keepdim=True) + 1e-12)
).cpu().numpy()

# ------------------------------------------------------------
# 可视化
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

for ax, vectors_plot, title in [
    (axes[0], initial_vectors / np.linalg.norm(initial_vectors, axis=1, keepdims=True),
     "Random initial bidirectional vectors"),
    (axes[1], final_vectors,
     "After paper-style $E_d + 10^{-3}E_g$ optimisation"),
]:
    ax.scatter(points_np[:, 0], points_np[:, 1], c="black", s=10, zorder=3)
    ax.quiver(
        points_np[:, 0], points_np[:, 1],
        vectors_plot[:, 0], vectors_plot[:, 1],
        color="crimson", scale=16, width=0.006,
    )
    ax.quiver(
        points_np[:, 0], points_np[:, 1],
        -vectors_plot[:, 0], -vectors_plot[:, 1],
        color="royalblue", scale=16, width=0.006,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

axes[2].plot(losses, label=r"$E_d+10^{-3}E_g$", color="black")
axes[2].plot(ed_history, label=r"$E_d$", color="tab:blue")
axes[2].plot(eg_history, label=r"$E_g$", color="tab:orange")
axes[2].set_yscale("log")
axes[2].set_title("Voronoi-bisector energy convergence")
axes[2].set_xlabel("Adam iteration")
axes[2].set_ylabel("energy (log scale)")
axes[2].legend()
axes[2].grid(alpha=0.25)

fig.suptitle("Paper-formula normal optimisation on ellipse", y=1.02)
fig.tight_layout()
fig.savefig(output_path, dpi=180, bbox_inches="tight")

np.savez(
    root / "outputs" / "15_paper_optimised_normals.npz",
    points=points_np,
    vectors=final_vectors,
    bisector_samples=samples_np,
    bisector_pairs=pair_ids_np,
)

print(f"Initial E_d={ed_history[0]:.6f}, E_g={eg_history[0]:.6f}")
print(f"Final   E_d={ed_history[-1]:.6f}, E_g={eg_history[-1]:.6f}")
print(f"Saved figure to: {output_path}")