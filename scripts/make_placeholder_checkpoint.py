"""Generate a placeholder ResNet50 checkpoint so the app can boot for local testing.

This creates a .pth file with random weights matching the architecture the
MalwareClassifier expects (ResNet50 with a 7-class final layer). It is NOT a
trained model — predictions will be meaningless — but it lets you exercise the
full scan pipeline and UI until the teammate's real checkpoint is available.

Usage:
    python scripts/make_placeholder_checkpoint.py
"""

from __future__ import annotations

import torch
from torchvision.models import resnet50

from app.components.classifier import MalwareClassifier
from app.config import AppSettings


def main() -> None:
    settings = AppSettings()
    num_classes = len(MalwareClassifier.CLASS_LABELS)

    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    out_path = settings.model_checkpoint_path
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    print(f"WROTE_PLACEHOLDER_CHECKPOINT {out_path}")


if __name__ == "__main__":
    main()
