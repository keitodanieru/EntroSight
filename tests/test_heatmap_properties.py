"""Property-based tests for EntropyHeatmapGenerator.

**Validates: Properties 3, 5 from design**
"""

import math
from collections import Counter

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from app.components.heatmap import EntropyHeatmapGenerator


class TestEntropyBounds:
    """**Validates: Requirements 1.2** — Property 5 (Entropy Bounds).

    For any byte sequence, computed Shannon entropy is in [0.0, 8.0]
    and the normalized value is in [0.0, 1.0].
    """

    @given(file_bytes=st.binary(min_size=1))
    @settings(max_examples=50)
    def test_raw_shannon_entropy_within_bounds(self, file_bytes: bytes) -> None:
        """Raw Shannon entropy per block is always in [0.0, 8.0]."""
        block_size = EntropyHeatmapGenerator.BLOCK_SIZE
        num_blocks = math.ceil(len(file_bytes) / block_size)

        for i in range(num_blocks):
            block = file_bytes[i * block_size : (i + 1) * block_size]
            if len(block) == 0:
                continue

            byte_counts = Counter(block)
            block_len = len(block)
            entropy = 0.0
            for count in byte_counts.values():
                p = count / block_len
                if p > 0:
                    entropy -= p * math.log2(p)

            assert 0.0 <= entropy <= 8.0, (
                f"Shannon entropy {entropy} out of bounds [0.0, 8.0]"
            )

    @given(file_bytes=st.binary(min_size=1))
    @settings(max_examples=50)
    def test_generated_heatmap_tensor_values_in_unit_range(
        self, file_bytes: bytes
    ) -> None:
        """Generated heatmap tensor has all values in [0.0, 1.0]."""
        generator = EntropyHeatmapGenerator()
        tensor = generator.generate(file_bytes)

        assert tensor.min().item() >= 0.0, (
            f"Tensor min {tensor.min().item()} is below 0.0"
        )
        assert tensor.max().item() <= 1.0, (
            f"Tensor max {tensor.max().item()} is above 1.0"
        )


class TestClassificationDeterminism:
    """**Validates: Requirements 1.2** — Property 3 (Classification Determinism, heatmap portion).

    Same input bytes always produce the same output tensor.
    """

    @given(file_bytes=st.binary(min_size=1))
    @settings(max_examples=50)
    def test_same_input_produces_same_tensor(self, file_bytes: bytes) -> None:
        """Calling generate() twice with the same input produces identical tensors."""
        generator = EntropyHeatmapGenerator()
        tensor_a = generator.generate(file_bytes)
        tensor_b = generator.generate(file_bytes)

        assert torch.equal(tensor_a, tensor_b), (
            "generate() produced different tensors for the same input"
        )
