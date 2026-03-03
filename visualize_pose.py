import json
import matplotlib.pyplot as plt
from pathlib import Path

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

POSE_DIR = Path("dataset_samples/pose")
N_CONSECUTIVE = 8  # number of consecutive frames to show per file


def draw_skeleton(ax, joints, title):
    for j1, j2 in SKELETON:
        if j1 in joints and j2 in joints:
            x1, y1 = joints[j1]['x'], joints[j1]['y']
            x2, y2 = joints[j2]['x'], joints[j2]['y']
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2)

    for jname, jval in joints.items():
        ax.plot(jval['x'], jval['y'], 'ro', markersize=5)

    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')


def load_valid_frames(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return [d for d in data if d['joints'] is not None]


def visualize_patient(patient_id):
    patient_dir = POSE_DIR / patient_id
    json_files = sorted(patient_dir.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found for {patient_id}")
        return

    for fpath in json_files:
        valid = load_valid_frames(fpath)
        if not valid:
            continue

        # Start from the middle of the sequence
        start = max(0, len(valid) // 2 - N_CONSECUTIVE // 2)
        frames = valid[start:start + N_CONSECUTIVE]

        label = f"{fpath.parent.name}/{fpath.stem.replace('_AlphaPose', '')}"
        fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 5))
        fig.suptitle(f'Pose Samples - {patient_id} | {label}', fontsize=14, fontweight='bold')

        for col, frame_data in enumerate(frames):
            draw_skeleton(axes[col], frame_data['joints'], f"Frame {frame_data['frame']}")

        plt.tight_layout()
        out_path = f"{patient_id}_{fpath.parent.name}_{fpath.stem.replace('_AlphaPose', '')}_consecutive.png"
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.show()


def main():
    patients = [p.name for p in POSE_DIR.iterdir() if p.is_dir()]
    if not patients:
        print(f"No patient folders found in {POSE_DIR}")
        return

    for patient_id in sorted(patients):
        visualize_patient(patient_id)


if __name__ == "__main__":
    main()
