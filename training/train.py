"""Training Script — Fine-tune ResNet50 on entropy heatmap dataset.

Usage:
    python training/train.py
    python training/train.py --device cuda
    python training/train.py --freeze-through layer3
    python training/train.py --resume training_outputs/best_model.pth
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import (
    AUGMENTATION,
    BATCH_SIZE,
    CLASS_LABELS,
    COSINE_T_0,
    COSINE_T_MULT,
    DATASET_DIR,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    FREEZE_THROUGH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    SEED,
    USE_AMP,
    WEIGHT_DECAY,
)


# ===========================================================================
# Dataset
# ===========================================================================


class HeatmapDataset(Dataset):
    """PyTorch Dataset for loading pre-generated heatmap tensors."""

    def __init__(self, root_dir: Path, transform=None):
        """Load all .pt tensor paths and their labels from directory structure.

        Args:
            root_dir: Path to train/ or val/ directory with class subfolders.
            transform: Optional torchvision transforms to apply.
        """
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        self.class_to_idx = {label: i for i, label in enumerate(CLASS_LABELS)}

        for label in CLASS_LABELS:
            class_dir = root_dir / label
            if not class_dir.exists():
                print(f"  WARNING: Class directory not found: {class_dir}")
                continue

            for pt_file in sorted(class_dir.glob("*.pt")):
                self.samples.append((pt_file, self.class_to_idx[label]))

        print(f"  Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        tensor = torch.load(path, weights_only=True)  # Shape: (3, 256, 256)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label

    def get_labels(self) -> list[int]:
        """Return all labels for computing class weights and sampler."""
        return [label for _, label in self.samples]


# ===========================================================================
# Early Stopping
# ===========================================================================


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Check if training should stop. Returns True if improved."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # Did not improve


# ===========================================================================
# Model Setup
# ===========================================================================


def create_model(freeze_through: str | None, device: torch.device) -> nn.Module:
    """Create ResNet50 with pretrained weights and frozen early layers.

    Args:
        freeze_through: Freeze all layers up to and including this layer.
                       Options: "layer1", "layer2", "layer3", or None.
        device: Device to place model on.

    Returns:
        Configured ResNet50 model.
    """
    # Load pretrained ImageNet weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Replace final fully-connected layer for our 7 classes
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # Freeze early layers
    if freeze_through:
        freeze_layers = ["conv1", "bn1", "layer1"]
        if freeze_through in ("layer2", "layer3"):
            freeze_layers.append("layer2")
        if freeze_through == "layer3":
            freeze_layers.append("layer3")

        frozen_params = 0
        for name, param in model.named_parameters():
            if any(name.startswith(layer) for layer in freeze_layers):
                param.requires_grad = False
                frozen_params += param.numel()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Frozen layers: {freeze_layers}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    model = model.to(device)
    return model


# ===========================================================================
# Training Loop
# ===========================================================================


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced loss."""
    class_counts = [0] * num_classes
    for label in labels:
        class_counts[label] += 1

    total = len(labels)
    weights = []
    for count in class_counts:
        if count > 0:
            weights.append(total / (num_classes * count))
        else:
            weights.append(1.0)

    # Normalize so weights average to 1.0
    avg_weight = sum(weights) / len(weights)
    weights = [w / avg_weight for w in weights]

    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    """Create a sampler that ensures balanced class representation per epoch."""
    class_counts = [0] * num_classes
    for label in labels:
        class_counts[label] += 1

    sample_weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    use_amp: bool,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Run validation. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    output_path: Path,
) -> None:
    """Save model checkpoint in the format expected by MalwareClassifier."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "class_labels": CLASS_LABELS,
        },
        output_path,
    )


# ===========================================================================
# Main
# ===========================================================================


def main(args: argparse.Namespace) -> None:
    """Main training entry point."""
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Data Transforms ===
    # Training: augmentation + normalization
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=AUGMENTATION["random_horizontal_flip"]),
        transforms.RandomRotation(degrees=AUGMENTATION["random_rotation_degrees"]),
        transforms.ColorJitter(
            brightness=AUGMENTATION["color_jitter_brightness"],
            contrast=AUGMENTATION["color_jitter_contrast"],
        ),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Validation: normalization only (no augmentation)
    val_transform = transforms.Compose([
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # === Load Datasets ===
    dataset_dir = Path(args.dataset_dir)
    print(f"\nLoading datasets from {dataset_dir}...")

    print("Training set:")
    train_dataset = HeatmapDataset(dataset_dir / "train", transform=train_transform)
    print("Validation set:")
    val_dataset = HeatmapDataset(dataset_dir / "val", transform=val_transform)

    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty. Run generate_dataset.py first.")
        sys.exit(1)

    # === Class Weights & Sampler ===
    train_labels = train_dataset.get_labels()
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
    sampler = create_weighted_sampler(train_labels, NUM_CLASSES)

    print(f"\nClass weights: {dict(zip(CLASS_LABELS, class_weights.tolist()))}")

    # === DataLoaders ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Weighted sampler for balanced batches
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # === Model ===
    print(f"\nCreating model (freeze through: {args.freeze_through})...")
    model = create_model(args.freeze_through, device)

    # === Loss, Optimizer, Scheduler ===
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=COSINE_T_0, T_mult=COSINE_T_MULT)

    # Mixed precision scaler
    use_amp = USE_AMP and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # === Resume from checkpoint ===
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"  Resumed at epoch {start_epoch}")

    # === Training Loop ===
    early_stopping = EarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
    )

    best_val_acc = 0.0
    log_path = output_dir / "training_log.csv"

    # CSV log header
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_sec"])

    print(f"\n{'='*70}")
    print(f"Starting training: {NUM_EPOCHS} epochs, batch_size={args.batch_size}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        print(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time])

        # Early stopping check
        improved = early_stopping.step(val_loss)

        if improved:
            # Save best model
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, output_dir / "best_model.pth")
            best_val_acc = val_acc
            print(f"  ↑ New best model saved (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")

        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1} (patience: {EARLY_STOP_PATIENCE})")
            break

    # === Final Summary ===
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")
    print(f"Training log: {log_path}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python training/evaluate.py")
    print(f"  2. Deploy:   copy {output_dir / 'best_model.pth'} → models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 malware classifier")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--freeze-through", default=FREEZE_THROUGH, help="Freeze layers: layer1, layer2, layer3, or none")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.freeze_through == "none":
        args.freeze_through = None

    main(args)
