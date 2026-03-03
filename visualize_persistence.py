import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import gudhi

SEGMENTATION_DIR = Path("dataset_samples/semantic_segmentation")
N_CONSECUTIVE = 8
N_POINTS = 150  # reduced density for Rips complex tractability
MAX_EDGE_LENGTH = 50  # max Rips filtration value (pixels)


def get_point_cloud(img_path, n_points=N_POINTS):
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.any(img > 10, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    if len(xs) > n_points:
        idx = np.random.choice(len(xs), n_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    return np.column_stack([xs, ys]).astype(float)


def plot_persistence_diagram(ax, dgm, title):
    colors = {0: 'steelblue', 1: 'tomato', 2: 'seagreen'}
    labels = {0: 'H0', 1: 'H1', 2: 'H2'}

    finite_pairs = [(dim, b, d) for dim, (b, d) in dgm if d != float('inf')]
    infinite_pairs = [(dim, b, d) for dim, (b, d) in dgm if d == float('inf')]

    if not finite_pairs and not infinite_pairs:
        ax.set_title(title, fontsize=8)
        return

    all_vals = [b for _, b, _ in finite_pairs] + [d for _, _, d in finite_pairs]
    all_vals += [b for _, b, _ in infinite_pairs]
    vmin = min(all_vals) if all_vals else 0
    vmax = max(all_vals) if all_vals else 1
    # Replace inf death with slightly above max for display
    inf_val = vmax * 1.1 + 5

    plotted = set()
    for dim, b, d in finite_pairs:
        label = labels.get(dim) if dim not in plotted else None
        ax.scatter(b, d, c=colors.get(dim, 'gray'), s=20, label=label, zorder=3)
        plotted.add(dim)

    for dim, b, _ in infinite_pairs:
        label = f"{labels.get(dim)} (∞)" if (dim + 10) not in plotted else None
        ax.scatter(b, inf_val, c=colors.get(dim, 'gray'), s=20,
                   marker='^', label=label, zorder=3)
        plotted.add(dim + 10)

    diag_max = max(vmax, inf_val) * 1.05
    ax.plot([vmin, diag_max], [vmin, diag_max], 'k--', linewidth=0.8, zorder=1)
    ax.axhline(inf_val, color='gray', linewidth=0.5, linestyle=':', zorder=1)
    ax.set_xlim(vmin - 1, diag_max)
    ax.set_ylim(vmin - 1, inf_val * 1.05)
    ax.set_xlabel("Birth", fontsize=7)
    ax.set_ylabel("Death", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    if plotted:
        ax.legend(fontsize=6, loc='lower right')


def visualize_trial(trial_dir):
    frames = sorted(trial_dir.glob("*.png"))
    if not frames:
        return

    start = max(0, len(frames) // 2 - N_CONSECUTIVE // 2)
    picks = frames[start:start + N_CONSECUTIVE]

    label = f"{trial_dir.parent.name}/{trial_dir.name}"
    patient_id = trial_dir.parent.parent.name

    fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 9))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Point Cloud & Persistence Diagram - {patient_id} | {label}',
                 fontsize=13, fontweight='bold')

    for col, fpath in enumerate(picks):
        pts = get_point_cloud(fpath)

        # --- Row 0: point cloud ---
        ax_pc = axes[0][col]
        if pts is not None:
            img = np.array(Image.open(fpath).convert("RGB"))
            colors = img[pts[:, 1].astype(int), pts[:, 0].astype(int)] / 255.0
            ax_pc.scatter(pts[:, 0], -pts[:, 1], c=colors, s=6, linewidths=0)
        ax_pc.set_facecolor('black')
        ax_pc.set_title(f"Frame {fpath.stem}", fontsize=9, color='black')
        ax_pc.set_aspect('equal')
        ax_pc.axis('off')

        # --- Row 1: persistence diagram ---
        ax_pd = axes[1][col]
        if pts is not None:
            rips = gudhi.RipsComplex(points=pts, max_edge_length=MAX_EDGE_LENGTH)
            st = rips.create_simplex_tree(max_dimension=2)
            st.compute_persistence()
            dgm = st.persistence()
            plot_persistence_diagram(ax_pd, dgm, f"PD Frame {fpath.stem}")
        else:
            ax_pd.set_title(f"Frame {fpath.stem} (no data)", fontsize=8)

    plt.tight_layout()
    out_path = f"{patient_id}_{trial_dir.parent.name}_{trial_dir.name}_persistence.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def main():
    np.random.seed(42)
    for patient_dir in sorted(SEGMENTATION_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        for trial_dir in sorted(patient_dir.rglob("*_DensePose")):
            if trial_dir.is_dir():
                visualize_trial(trial_dir)


if __name__ == "__main__":
    main()
