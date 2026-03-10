import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
import gudhi

POSE_DIR = Path("dataset_samples/pose")


def load_valid_frames(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return [d for d in data if d['joints'] is not None]


def joints_to_points(joints):
    return np.array([[j['x'], j['y']] for j in joints.values()])


def compute_persistence(pts):
    rips = gudhi.RipsComplex(points=pts)
    st = rips.create_simplex_tree(max_dimension=2)
    dgm = st.persistence()
    return dgm


def extract_pairs(dgm, dim):
    """Return finite (birth, death) pairs for a given homology dimension."""
    return np.array([(b, d) for d_i, (b, d) in dgm if d_i == dim and d != float('inf')])


def match_pairs(pairs_a, pairs_b):
    """
    Match persistence pairs between two consecutive frames using the
    Hungarian algorithm (minimises L2 distance in birth-death plane).
    Unpaired points are matched to their diagonal projection (b, b).
    Returns list of ((b0,d0), (b1,d1)) matched vines.
    """
    if len(pairs_a) == 0 or len(pairs_b) == 0:
        return []

    # Pad the smaller set with diagonal projections so sizes match
    n, m = len(pairs_a), len(pairs_b)
    size = max(n, m)

    def pad_to_diagonal(pairs, target):
        if len(pairs) < target:
            diag = np.array([[(b + d) / 2, (b + d) / 2] for b, d in pairs])
            padding = diag[:target - len(pairs)] if len(diag) > 0 else np.zeros((target - len(pairs), 2))
            # pad with midpoints of existing pairs cycled
            extra = np.tile(diag, (target, 1))[:target - len(pairs)]
            return np.vstack([pairs, extra])
        return pairs

    a_padded = pad_to_diagonal(pairs_a, size)
    b_padded = pad_to_diagonal(pairs_b, size)

    # Cost matrix: L2 distance between every pair of points
    diff = a_padded[:, None, :] - b_padded[None, :, :]
    cost = np.sqrt((diff ** 2).sum(axis=2))

    row_ind, col_ind = linear_sum_assignment(cost)

    vines = []
    for i, j in zip(row_ind, col_ind):
        if i < n and j < m:
            vines.append((pairs_a[i], pairs_b[j]))
    return vines


def draw_vineyard(ax, all_pairs_per_frame, frame_indices, color):
    """
    all_pairs_per_frame: list of np.arrays of shape (k, 2) — one per frame
    frame_indices: corresponding frame numbers (compressed to 0..1 range)
    """
    # Compress frame indices to [0, 1] so slices are tightly packed
    t_min, t_max = min(frame_indices), max(frame_indices)
    t_norm = [(t - t_min) / (t_max - t_min) if t_max > t_min else 0 for t in frame_indices]

    for pairs, t in zip(all_pairs_per_frame, t_norm):
        if len(pairs):
            ax.scatter(pairs[:, 0], pairs[:, 1], t,
                       c=color, s=30, alpha=0.5, depthshade=False, zorder=3)


def visualize_trial_vineyard(json_path):
    valid = load_valid_frames(json_path)
    if not valid:
        return

    label = f"{json_path.parent.name}/{json_path.stem.replace('_AlphaPose', '')}"
    patient_id = json_path.parent.parent.name

    h0_per_frame, h1_per_frame, frame_indices = [], [], []

    for frame_data in valid:
        pts = joints_to_points(frame_data['joints'])
        dgm = compute_persistence(pts)
        h0 = extract_pairs(dgm, 0)
        h1 = extract_pairs(dgm, 1)
        h0_per_frame.append(h0 if len(h0) else np.empty((0, 2)))
        h1_per_frame.append(h1 if len(h1) else np.empty((0, 2)))
        frame_indices.append(frame_data['frame'])

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f'Vineyard - {patient_id} | {label}', fontsize=14, fontweight='bold')

    for subplot_idx, (dim_label, pairs_per_frame, color) in enumerate([
        ('H0 (Components)', h0_per_frame, 'steelblue'),
        ('H1 (Loops)',      h1_per_frame, 'tomato'),
    ]):
        ax = fig.add_subplot(1, 2, subplot_idx + 1, projection='3d')
        draw_vineyard(ax, pairs_per_frame, frame_indices, color)

        # Diagonal plane
        all_vals = np.concatenate([p for p in pairs_per_frame if len(p)])
        if len(all_vals):
            vmin, vmax = all_vals.min(), all_vals.max()
            grid = np.linspace(vmin, vmax, 10)
            B, T = np.meshgrid(grid, np.linspace(0, 1, 10))
            ax.plot_surface(B, B, T, alpha=0.07, color='gray')

        ax.set_xlabel("Birth", fontsize=9, labelpad=6)
        ax.set_ylabel("Death", fontsize=9, labelpad=6)
        ax.set_zlabel("Time", fontsize=9, labelpad=6)
        ax.set_title(dim_label, fontsize=11)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=25, azim=-55)

    plt.tight_layout()
    out_path = f"{patient_id}_{json_path.parent.name}_{json_path.stem.replace('_AlphaPose', '')}_vineyard.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    for patient_dir in sorted(POSE_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        for json_path in sorted(patient_dir.rglob("*.json")):
            visualize_trial_vineyard(json_path)


if __name__ == "__main__":
    main()
