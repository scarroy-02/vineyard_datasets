import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

SEGMENTATION_DIR = Path("dataset_samples/semantic_segmentation")
N_CONSECUTIVE = 8


def visualize_trial(trial_dir):
    frames = sorted(trial_dir.glob("*.png"))
    if not frames:
        return

    start = max(0, len(frames) // 2 - N_CONSECUTIVE // 2)
    picks = frames[start:start + N_CONSECUTIVE]

    label = f"{trial_dir.parent.name}/{trial_dir.name}"
    patient_id = trial_dir.parent.parent.name

    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 5))
    fig.suptitle(f'Segmentation Samples - {patient_id} | {label}', fontsize=14, fontweight='bold')

    for ax, fpath in zip(axes, picks):
        img = mpimg.imread(fpath)
        ax.imshow(img)
        ax.set_title(f"Frame {fpath.stem}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    out_path = f"{patient_id}_{trial_dir.parent.name}_{trial_dir.name}_consecutive.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def main():
    for patient_dir in sorted(SEGMENTATION_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        for trial_dir in sorted(patient_dir.rglob("*_DensePose")):
            if trial_dir.is_dir():
                visualize_trial(trial_dir)


if __name__ == "__main__":
    main()
