"""Evaluation Script — generate metrics, confusion matrix, and per-class report.

Usage:
    python training/evaluate.py
    python training/evaluate.py --checkpoint training_outputs/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import (
    BATCH_SIZE,
    CLASS_LABELS,
    DATASET_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    NUM_WORKERS,
    OUTPUT_DIR,
)
from training.train import HeatmapDataset


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_accuracy", "?")
    print(f"Loaded checkpoint from epoch {epoch} (val_acc: {val_acc})")

    return model


@torch.no_grad()
def get_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[list[float]]]:
    """Run inference on entire dataset. Returns (true_labels, pred_labels, all_probs)."""
    all_true: list[int] = []
    all_pred: list[int] = []
    all_probs: list[list[float]] = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)

        _, predicted = outputs.max(1)
        all_true.extend(targets.tolist())
        all_pred.extend(predicted.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_true, all_pred, all_probs


def compute_metrics(true_labels: list[int], pred_labels: list[int]) -> dict:
    """Compute per-class precision, recall, F1, and overall accuracy."""
    metrics = {}

    # Overall accuracy
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    metrics["accuracy"] = correct / len(true_labels)

    # Per-class metrics
    per_class = {}
    for cls_idx, cls_name in enumerate(CLASS_LABELS):
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls_idx and p == cls_idx)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls_idx and p == cls_idx)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls_idx and p != cls_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for t in true_labels if t == cls_idx)

        per_class[cls_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    metrics["per_class"] = per_class

    # Macro averages
    metrics["macro_precision"] = np.mean([v["precision"] for v in per_class.values()])
    metrics["macro_recall"] = np.mean([v["recall"] for v in per_class.values()])
    metrics["macro_f1"] = np.mean([v["f1"] for v in per_class.values()])

    return metrics


def compute_confusion_matrix(true_labels: list[int], pred_labels: list[int]) -> np.ndarray:
    """Compute NxN confusion matrix."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        matrix[t][p] += 1
    return matrix


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    """Save confusion matrix as a PNG plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax)

    # Labels
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_LABELS, fontsize=9)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Annotate cells with counts
    thresh = matrix.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, format(matrix[i, j], "d"),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def print_classification_report(metrics: dict) -> None:
    """Print formatted classification report to stdout."""
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION REPORT")
    print(f"{'='*70}")
    print(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*60}")

    for cls_name, cls_metrics in metrics["per_class"].items():
        print(
            f"{cls_name:<20} "
            f"{cls_metrics['precision']:>10.4f} "
            f"{cls_metrics['recall']:>10.4f} "
            f"{cls_metrics['f1']:>10.4f} "
            f"{cls_metrics['support']:>10d}"
        )

    print(f"{'-'*60}")
    print(
        f"{'Macro Avg':<20} "
        f"{metrics['macro_precision']:>10.4f} "
        f"{metrics['macro_recall']:>10.4f} "
        f"{metrics['macro_f1']:>10.4f}"
    )
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")

    # Highlight potential false positive issue
    benign_metrics = metrics["per_class"].get("Benign", {})
    if benign_metrics:
        print(f"\n--- False Positive Analysis ---")
        print(f"Benign Precision: {benign_metrics['precision']:.4f} (low = other classes wrongly predict Benign)")
        print(f"Benign Recall:    {benign_metrics['recall']:.4f} (low = Benign files misclassified as malware)")
        if benign_metrics["recall"] < 0.90:
            print(f"⚠️  WARNING: Benign recall below 90% — high false positive rate!")


def main(args: argparse.Namespace) -> None:
    """Main evaluation entry point."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = load_model(checkpoint_path, device)

    # Load validation dataset
    val_transform = transforms.Compose([
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset_dir = Path(args.dataset_dir)
    print(f"\nLoading validation set from {dataset_dir / 'val'}...")
    val_dataset = HeatmapDataset(dataset_dir / "val", transform=val_transform)

    if len(val_dataset) == 0:
        print("ERROR: Validation dataset is empty.")
        sys.exit(1)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # Run predictions
    print("Running inference on validation set...")
    true_labels, pred_labels, all_probs = get_predictions(model, val_loader, device)

    # Compute metrics
    metrics = compute_metrics(true_labels, pred_labels)
    print_classification_report(metrics)

    # Confusion matrix
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = compute_confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    # Print raw confusion matrix for Benign row (false positive detail)
    benign_idx = CLASS_LABELS.index("Benign")
    print(f"\nBenign row in confusion matrix (where benign files get classified):")
    for j, label in enumerate(CLASS_LABELS):
        count = cm[benign_idx][j]
        if count > 0:
            print(f"  → predicted as {label}: {count}")

    print(f"\nEvaluation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained malware classifier")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUT_DIR / "best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args)
