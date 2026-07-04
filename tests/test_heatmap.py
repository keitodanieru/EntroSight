"""Unit tests for EntropyHeatmapGenerator.

Tests cover output shape, entropy behavior for known inputs,
and PNG visualization output.
"""

import os

import torch

from app.components.heatmap import EntropyHeatmapGenerator


class TestGenerateOutputShape:
    """Test that generate() returns the correct tensor shape."""

    def test_output_shape_is_3_256_256(self) -> None:
        """Output tensor shape must be (3, 256, 256) for ResNet50 input."""
        generator = EntropyHeatmapGenerator()
        # Use a small PE-like file (MZ header + padding)
        file_bytes = b"MZ" + b"\x00" * 1024
        tensor = generator.generate(file_bytes)

        assert tensor.shape == (3, 256, 256)

    def test_output_shape_with_large_input(self) -> None:
        """Shape stays (3, 256, 256) regardless of input size."""
        generator = EntropyHeatmapGenerator()
        file_bytes = os.urandom(100_000)
        tensor = generator.generate(file_bytes)

        assert tensor.shape == (3, 256, 256)


class TestAllZeroBytesLowEntropy:
    """Test that all-zero bytes produce low entropy values."""

    def test_zero_bytes_produce_low_entropy(self) -> None:
        """A file of all zeros has zero Shannon entropy (single byte value)."""
        generator = EntropyHeatmapGenerator()
        file_bytes = b"\x00" * 4096
        tensor = generator.generate(file_bytes)

        # All-zero blocks have entropy = 0 (only one unique byte value)
        # After normalization by 8.0, values should be 0.0
        assert tensor.max().item() == 0.0

    def test_repeated_single_byte_produces_zero_entropy(self) -> None:
        """Any single repeated byte value yields zero entropy."""
        generator = EntropyHeatmapGenerator()
        file_bytes = b"\xAB" * 2048
        tensor = generator.generate(file_bytes)

        assert tensor.max().item() == 0.0


class TestRandomBytesHighEntropy:
    """Test that random bytes produce high entropy values."""

    def test_random_bytes_produce_high_entropy(self) -> None:
        """Uniformly random bytes should yield entropy close to 1.0 (normalized)."""
        generator = EntropyHeatmapGenerator()
        # Use a large random input so blocks are full and entropy is high
        # os.urandom produces cryptographically random bytes
        file_bytes = os.urandom(256 * 256)  # 256 full blocks
        tensor = generator.generate(file_bytes)

        # Random bytes should produce high entropy (close to 8.0/8.0 = 1.0)
        # Allow some tolerance since finite blocks won't be perfectly uniform
        mean_value = tensor.mean().item()
        assert mean_value > 0.85, (
            f"Expected high entropy for random bytes, got mean={mean_value:.4f}"
        )


class TestGenerateVisualization:
    """Test that generate_visualization() returns valid PNG bytes."""

    def test_returns_png_bytes(self) -> None:
        """Visualization output must be valid PNG (starts with PNG signature)."""
        generator = EntropyHeatmapGenerator()
        file_bytes = b"MZ" + os.urandom(2048)
        png_bytes = generator.generate_visualization(file_bytes)

        # PNG files start with an 8-byte signature
        png_signature = b"\x89PNG\r\n\x1a\n"
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes[:8] == png_signature

    def test_png_bytes_are_non_trivial(self) -> None:
        """PNG output should have a reasonable size (not empty/corrupt)."""
        generator = EntropyHeatmapGenerator()
        file_bytes = os.urandom(4096)
        png_bytes = generator.generate_visualization(file_bytes)

        # A 256x256 RGB PNG should be at least a few KB
        assert len(png_bytes) > 1000
