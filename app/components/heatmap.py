"""Entropy Heatmap Generator — converts PE bytes to byte-entropy heatmap tensors."""

import io
import math
from collections import Counter

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")  # Non-interactive backend for PNG generation


class EntropyHeatmapGenerator:
    """Converts raw PE file bytes into a 256x256 byte-entropy heatmap tensor.

    The heatmap captures the spatial distribution of byte randomness across
    the file, which serves as input for the ResNet50 malware classifier.
    """

    BLOCK_SIZE: int = 256
    IMAGE_SIZE: tuple[int, int] = (256, 256)

    def generate(self, file_bytes: bytes) -> torch.Tensor:
        """Convert PE bytes to normalized entropy heatmap tensor.

        Algorithm:
        1. Divide file into fixed-size blocks
        2. Compute Shannon entropy for each block
        3. Arrange entropy values into a 2D grid
        4. Resize to 256x256 via bilinear interpolation
        5. Replicate to 3 channels (RGB) for ResNet50 input

        Args:
            file_bytes: Raw bytes of the PE file.

        Returns:
            Tensor of shape (3, 256, 256) with values in [0.0, 1.0].
        """
        # Step 1: Divide into blocks
        num_blocks = math.ceil(len(file_bytes) / self.BLOCK_SIZE)
        entropy_values: list[float] = []

        # Step 2: Compute Shannon entropy per block
        for i in range(num_blocks):
            block = file_bytes[i * self.BLOCK_SIZE : (i + 1) * self.BLOCK_SIZE]
            if len(block) == 0:
                entropy_values.append(0.0)
                continue

            # Shannon entropy: H = -Σ p(x) * log2(p(x))
            byte_counts = Counter(block)
            block_len = len(block)
            entropy = 0.0
            for count in byte_counts.values():
                p = count / block_len
                if p > 0:
                    entropy -= p * math.log2(p)

            # Normalize to [0, 1] range (max entropy for bytes = 8.0)
            entropy_values.append(entropy / 8.0)

        # Step 3: Arrange into 2D grid (square, row-major)
        grid_side = math.ceil(math.sqrt(num_blocks))
        # Pad with zeros if needed
        while len(entropy_values) < grid_side * grid_side:
            entropy_values.append(0.0)

        grid = torch.tensor(
            entropy_values[: grid_side * grid_side], dtype=torch.float32
        ).reshape(grid_side, grid_side)

        # Step 4: Resize to 256x256 via bilinear interpolation
        grid_resized = F.interpolate(
            grid.unsqueeze(0).unsqueeze(0),
            size=self.IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Clamp to ensure values stay within [0, 1] after interpolation
        grid_resized = grid_resized.clamp(0.0, 1.0)

        # Step 5: Replicate to 3 channels (RGB) for ResNet50
        heatmap_tensor = grid_resized.unsqueeze(0).repeat(3, 1, 1)

        return heatmap_tensor

    def generate_visualization(self, file_bytes: bytes) -> bytes:
        """Generate PNG visualization of the entropy heatmap.

        Applies a matplotlib colormap (inferno) to the entropy values
        and returns the result as PNG bytes suitable for display in the UI.

        Args:
            file_bytes: Raw bytes of the PE file.

        Returns:
            PNG image bytes.
        """
        # Generate the heatmap tensor (take single channel)
        heatmap_tensor = self.generate(file_bytes)
        heatmap_2d = heatmap_tensor[0].numpy()  # Shape: (256, 256)

        # Apply colormap (inferno works well for entropy visualization)
        colormap = matplotlib.colormaps["inferno"]
        colored = colormap(heatmap_2d)  # Returns RGBA array (256, 256, 4)

        # Convert to uint8 RGB image
        rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)
        image = Image.fromarray(rgb_array, mode="RGB")

        # Save to PNG bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
