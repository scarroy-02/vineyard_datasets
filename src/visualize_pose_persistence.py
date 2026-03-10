import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import gudhi

POSE_DIR = Path("dataset_samples/pose")
N_CONSECUTIVE = 8

SKELETON = [
    ('nose', 'l_eye'), ('nose', 'r_eye'),
    ('l_eye', 'l_ear'), ('r_eye', 'r_ear'),
    ('l_shoulder', 'r_shoulder'),
    ('l_shoulder', 'l_elbow'), ('r_shoulder', 'r_elbow'),
    ('l_elbow', 'l_wrist'), ('r_elbow', 'r_wrist'),
    ('l_shoulder', 'l_hip'), ('r_shoulder', 'r_hip'),
    ('l_hip', 'r_hip'),
    ('l_hip', 'l_knee'), ('r_hip', 'r_knee'),
    ('l_knee', 'l_ankle'), ('r_knee', 'r_ankle'),
]


def load_valid_frames(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return [d for d in data if d['joints'] is not None]


def joints_to_points(joints):
    return np.array([[j['x'], j['y']] for j in joints.values()])


def draw_skeleton(ax, joints, title):
    pts = joints_to_points(joints)
    for j1, j2 in SKELETON:
        if j1 in joints and j2 in joints:
            x1, y1 = joints[j1]['x'], joints[j1]['y']
            x2, y2 = joints[j2]['x'], joints[j2]['y']
            ax.plot([x1, x2], [-y1, -y2], 'b-', linewidth=1.5)
    ax.scatter(pts[:, 0], -pts[:, 1], c='red', s=20, zorder=3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=8)


def plot_persistence_diagram(ax, dgm, title):
    colors = {0: 'steelblue', 1: 'tomato', 2: 'seagreen'}
    labels = {0: 'H0', 1: 'H1', 2: 'H2'}

    finite_pairs = [(dim, b, d) for dim, (b, d) in dgm if d != float('inf')]
    infinite_pairs = [(dim, b, d) for dim, (b, d) in dgm if d == float('inf')]

    all_vals = [b for _, b, _ in finite_pairs] + [d for _, _, d in finite_pairs]
    all_vals += [b for _, b, _ in infinite_pairs]
    if not all_vals:
        ax.set_title(title, fontsize=8)
        return

    vmin, vmax = min(all_vals), max(all_vals)
    inf_val = vmax * 1.15 + 5

    plotted = set()
    for dim, b, d in finite_pairs:
        label = labels.get(dim) if dim not in plotted else None
        ax.scatter(b, d, c=colors.get(dim, 'gray'), s=25, label=label, zorder=3)
        plotted.add(dim)

    for dim, b, _ in infinite_pairs:
        key = dim + 10
        label = f"{labels.get(dim)} (∞)" if key not in plotted else None
        ax.scatter(b, inf_val, c=colors.get(dim, 'gray'), s=25,
                   marker='^', label=label, zorder=3)
        plotted.add(key)

    diag_max = max(vmax, inf_val) * 1.05
    ax.plot([vmin, diag_max], [vmin, diag_max], 'k--', linewidth=0.8)
    ax.axhline(inf_val, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlim(vmin - 2, diag_max)
    ax.set_ylim(vmin - 2, inf_val * 1.05)
    ax.set_xlabel("Birth", fontsize=7)
    ax.set_ylabel("Death", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)
    if plotted:
        ax.legend(fontsize=6, loc='lower right')


def visualize_trial(json_path):
    valid = load_valid_frames(json_path)
    if not valid:
        return

    start = max(0, len(valid) // 2 - N_CONSECUTIVE // 2)
    picks = valid[start:start + N_CONSECUTIVE]

    label = f"{json_path.parent.name}/{json_path.stem.replace('_AlphaPose', '')}"
    patient_id = json_path.parent.parent.name

    fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 9))
    fig.suptitle(f'Pose & Persistence Diagram - {patient_id} | {label}',
                 fontsize=13, fontweight='bold')

    for col, frame_data in enumerate(picks):
        joints = frame_data['joints']
        pts = joints_to_points(joints)

        # Row 0: skeleton
        draw_skeleton(axes[0][col], joints, f"Frame {frame_data['frame']}")

        # Row 1: persistence diagram via Rips on joint positions
        rips = gudhi.RipsComplex(points=pts)
        st = rips.create_simplex_tree(max_dimension=2)
        dgm = st.persistence()
        plot_persistence_diagram(axes[1][col], dgm, f"PD Frame {frame_data['frame']}")

    plt.tight_layout()
    out_path = f"{patient_id}_{json_path.parent.name}_{json_path.stem.replace('_AlphaPose', '')}_persistence.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def main():
    for patient_dir in sorted(POSE_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        for json_path in sorted(patient_dir.rglob("*.json")):
            visualize_trial(json_path)


if __name__ == "__main__":
    main()
