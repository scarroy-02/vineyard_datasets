import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

SEGMENTATION_DIR = Path("dataset_samples/semantic_segmentation")
N_CONSECUTIVE = 8
N_POINTS = 800  # points sampled per frame


def get_point_cloud(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    # Mask of non-black pixels (human region)
    mask = np.any(img > 10, axis=2)
    ys, xs = np.where(mask)
    colors = img[ys, xs] / 255.0

    # Subsample
    if len(xs) > N_POINTS:
        idx = np.random.choice(len(xs), N_POINTS, replace=False)
        xs, ys, colors = xs[idx], ys[idx], colors[idx]

    return xs, ys, colors


def visualize_trial(trial_dir):
    frames = sorted(trial_dir.glob("*.png"))
    if not frames:
        return

    start = max(0, len(frames) // 2 - N_CONSECUTIVE // 2)
    picks = frames[start:start + N_CONSECUTIVE]

    label = f"{trial_dir.parent.name}/{trial_dir.name}"
    patient_id = trial_dir.parent.parent.name

    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 5))
    fig.patch.set_facecolor('black')
    fig.suptitle(f'Segmentation Point Cloud - {patient_id} | {label}', fontsize=14,
                 fontweight='bold', color='white')

    for ax, fpath in zip(axes, picks):
        xs, ys, colors = get_point_cloud(fpath)
        ax.scatter(xs, -ys, c=colors, s=4, linewidths=0)
        ax.set_title(f"Frame {fpath.stem}", fontsize=9, color='white')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    out_path = f"{patient_id}_{trial_dir.parent.name}_{trial_dir.name}_pointcloud.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='black')
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
