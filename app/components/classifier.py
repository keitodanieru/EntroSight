"""MalwareClassifier — ResNet50 inference for malware family classification."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50

from app.models import ClassificationResult


class MalwareClassifier:
    """Classifies byte-entropy heatmaps into malware families using ResNet50."""

    CLASS_LABELS: list[str] = [
        "AgentTesla",
        "Remcos",
        "DCRat",
        "FormBook",
        "RedLine",
        "AsyncRAT",
        "Benign",
    ]

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load model checkpoint and prepare for inference.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            device: Device to run inference on (default: "cpu").
        """
        self.device = torch.device(device)
        num_classes = len(self.CLASS_LABELS)

        # Initialize ResNet50 architecture with modified final layer
        self.model = resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        # Load trained weights from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Set to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)

        # ImageNet normalization transform
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def classify(self, heatmap_tensor: torch.Tensor) -> ClassificationResult:
        """Run inference on a single heatmap tensor.

        Args:
            heatmap_tensor: Tensor of shape (3, 256, 256) with values in [0, 1].

        Returns:
            ClassificationResult with label, confidence, all probabilities, and timing.
        """
        start_time = time.perf_counter()

        # Step 1: Apply ImageNet normalization
        normalized = self._normalize(heatmap_tensor)

        # Step 2: Add batch dimension -> (1, 3, 256, 256)
        input_tensor = normalized.unsqueeze(0).to(self.device)

        # Step 3: Forward pass (no gradient computation)
        with torch.no_grad():
            logits = self.model(input_tensor)

        # Step 4: Softmax for probabilities
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

        # Step 5: Extract top prediction
        top_idx = torch.argmax(probabilities).item()
        confidence = probabilities[top_idx].item()
        predicted_label = self.CLASS_LABELS[top_idx]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ClassificationResult(
            predicted_label=predicted_label,
            confidence=confidence,
            all_probabilities={
                label: prob.item()
                for label, prob in zip(self.CLASS_LABELS, probabilities)
            },
            inference_time_ms=elapsed_ms,
        )

    def get_grad_cam(self, heatmap_tensor: torch.Tensor) -> torch.Tensor:
        """Generate Grad-CAM activation map for explainability.

        Uses the last convolutional layer (layer4) of ResNet50 to produce
        a class-discriminative localization map highlighting important regions.

        Args:
            heatmap_tensor: Tensor of shape (3, 256, 256) with values in [0, 1].

        Returns:
            Tensor of shape (256, 256) with activation intensities in [0, 1].
        """
        # Prepare input
        normalized = self._normalize(heatmap_tensor)
        input_tensor = normalized.unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        # Storage for hook outputs
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        # Register hooks on the last conv layer (layer4)
        target_layer = self.model.layer4

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            logits = self.model(input_tensor)

            # Get the predicted class
            top_idx = torch.argmax(logits, dim=1).item()

            # Backward pass for the predicted class
            self.model.zero_grad()
            target_score = logits[0, top_idx]
            target_score.backward()

            # Compute Grad-CAM
            # Global average pooling of gradients -> channel weights
            grad = gradients[0]  # Shape: (1, C, H, W)
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of activation maps
            activation = activations[0]  # Shape: (1, C, H, W)
            cam = torch.sum(weights * activation, dim=1, keepdim=True)  # (1, 1, H, W)

            # Apply ReLU (only positive contributions)
            cam = F.relu(cam)

            # Resize to input dimensions (256x256)
            cam = F.interpolate(cam, size=(256, 256), mode="bilinear", align_corners=False)
            cam = cam.squeeze()  # Shape: (256, 256)

            # Normalize to [0, 1]
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min > 0:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros(256, 256)

        finally:
            # Remove hooks
            fwd_handle.remove()
            bwd_handle.remove()

        return cam.detach()
