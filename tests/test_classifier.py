"""Unit tests for MalwareClassifier component."""

import tempfile
from pathlib import Path

import pytest
import torch
from torchvision.models import resnet50

from app.components.classifier import MalwareClassifier
from app.models import ClassificationResult


@pytest.fixture
def mock_checkpoint_path(tmp_path: Path) -> str:
    """Create a temporary .pth checkpoint with random weights in the correct format."""
    # Build a ResNet50 with the modified final layer (7 classes)
    num_classes = 7
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Save checkpoint in the expected format
    checkpoint = {"model_state_dict": model.state_dict()}
    checkpoint_file = tmp_path / "test_model.pth"
    torch.save(checkpoint, checkpoint_file)

    return str(checkpoint_file)


@pytest.fixture
def classifier(mock_checkpoint_path: str) -> MalwareClassifier:
    """Instantiate a MalwareClassifier with the mock checkpoint."""
    return MalwareClassifier(checkpoint_path=mock_checkpoint_path, device="cpu")


@pytest.fixture
def sample_heatmap() -> torch.Tensor:
    """Generate a random heatmap tensor of the expected input shape (3, 256, 256)."""
    return torch.rand(3, 256, 256)


# ---------------------------------------------------------------------------
# Test: Loading a valid checkpoint doesn't raise
# ---------------------------------------------------------------------------


class TestMalwareClassifierLoading:
    """Checkpoint loading tests."""

    def test_loads_valid_checkpoint_without_error(self, mock_checkpoint_path: str) -> None:
        """Loading a valid checkpoint should not raise any exceptions."""
        classifier = MalwareClassifier(checkpoint_path=mock_checkpoint_path, device="cpu")
        assert classifier is not None

    def test_model_is_in_eval_mode(self, classifier: MalwareClassifier) -> None:
        """After loading, model should be in eval mode (not training)."""
        assert not classifier.model.training


# ---------------------------------------------------------------------------
# Test: classify() returns a valid ClassificationResult
# ---------------------------------------------------------------------------


class TestMalwareClassifierClassify:
    """Classification inference tests."""

    def test_returns_classification_result(
        self, classifier: MalwareClassifier, sample_heatmap: torch.Tensor
    ) -> None:
        """classify() should return a ClassificationResult instance."""
        result = classifier.classify(sample_heatmap)
        assert isinstance(result, ClassificationResult)

    def test_probabilities_sum_to_approximately_one(
        self, classifier: MalwareClassifier, sample_heatmap: torch.Tensor
    ) -> None:
        """All probabilities from softmax should sum to ~1.0."""
        result = classifier.classify(sample_heatmap)
        total = sum(result.all_probabilities.values())
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected ~1.0"

    def test_predicted_label_is_valid_class(
        self, classifier: MalwareClassifier, sample_heatmap: torch.Tensor
    ) -> None:
        """predicted_label must be one of the defined CLASS_LABELS or 'Unknown'."""
        result = classifier.classify(sample_heatmap)
        valid_labels = set(MalwareClassifier.CLASS_LABELS) | {MalwareClassifier.UNKNOWN_LABEL}
        assert result.predicted_label in valid_labels

    def test_confidence_between_zero_and_one(
        self, classifier: MalwareClassifier, sample_heatmap: torch.Tensor
    ) -> None:
        """confidence should be in the range (0.0, 1.0]."""
        result = classifier.classify(sample_heatmap)
        assert 0.0 < result.confidence <= 1.0

    def test_inference_time_is_positive(
        self, classifier: MalwareClassifier, sample_heatmap: torch.Tensor
    ) -> None:
        """inference_time_ms must be a positive value."""
        result = classifier.classify(sample_heatmap)
        assert result.inference_time_ms > 0.0

    def test_all_probabilities_keys_match_class_labels(
        self, classifier: MalwareClassifier, sample_heatmap: torch.Tensor
    ) -> None:
        """all_probabilities dict should have exactly the CLASS_LABELS as keys."""
        result = classifier.classify(sample_heatmap)
        assert set(result.all_probabilities.keys()) == set(MalwareClassifier.CLASS_LABELS)

    def test_all_probabilities_values_are_non_negative(
        self, classifier: MalwareClassifier, sample_heatmap: torch.Tensor
    ) -> None:
        """Each probability value should be non-negative."""
        result = classifier.classify(sample_heatmap)
        for label, prob in result.all_probabilities.items():
            assert prob >= 0.0, f"Probability for {label} is negative: {prob}"


# ---------------------------------------------------------------------------
# Test: get_grad_cam() returns correct shape and range
# ---------------------------------------------------------------------------


class TestMalwareClassifierGradCam:
    """Grad-CAM explainability tests."""

    def test_grad_cam_returns_correct_shape(
        self, classifier: MalwareClassifier, sample_heatmap: torch.Tensor
    ) -> None:
        """get_grad_cam() should return a tensor of shape (256, 256)."""
        cam = classifier.get_grad_cam(sample_heatmap)
        assert cam.shape == (256, 256)

    def test_grad_cam_values_in_zero_one_range(
        self, classifier: MalwareClassifier, sample_heatmap: torch.Tensor
    ) -> None:
        """Grad-CAM output values should be normalized to [0, 1]."""
        cam = classifier.get_grad_cam(sample_heatmap)
        assert cam.min().item() >= 0.0
        assert cam.max().item() <= 1.0
