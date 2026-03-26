"""
models/attention.py — v2
LayerCAM multi-scale attention generator.

Contract (from v2_implementation_plan.md §4.4):
  - Library: pytorch-grad-cam LayerCAM
  - Target layers: block3, block6, block12
  - Fusion weights: [0.2, 0.3, 0.5] early→late
  - Output: (H, W) float32 heatmap, values in [0, 1]
  - Unchanged from v1 (this module was correct)
"""

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class LayerCAMGenerator:
    """
    Generates a fused multi-scale LayerCAM attention map from an XceptionNet model.

    Fusion strategy (per v2 plan §4.4):
      - Compute LayerCAM for each of block3, block6, block12 separately
      - Upsample each to output_size
      - Weighted sum with weights [0.2, 0.3, 0.5]
      - Min-max normalise to [0, 1]
    """

    FUSION_WEIGHTS = [0.2, 0.3, 0.5]  # early → late

    def __init__(self, model, output_size: tuple[int, int] = (384, 384)):
        """
        Args:
            model:       XceptionNetClassifier instance (must be in eval mode).
            output_size: (H, W) for the output heatmap.
        """
        self.model = model
        self.output_size = output_size
        self.target_layers = model.get_target_layers()  # [block3, block6, block12]

        if len(self.target_layers) != len(self.FUSION_WEIGHTS):
            raise ValueError(
                f"Expected {len(self.FUSION_WEIGHTS)} target layers, "
                f"got {len(self.target_layers)}"
            )

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Generate a fused LayerCAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) normalised float tensor (requires grad).
            target_class: Class index for CAM — 1 = fake (default), 0 = real.

        Returns:
            heatmap: (H, W) float32 ndarray in [0, 1].
        """
        targets = [ClassifierOutputTarget(target_class)]
        fused = np.zeros(self.output_size, dtype=np.float32)

        # Ensure gradients flow — required by pytorch-grad-cam
        input_tensor = input_tensor.detach().requires_grad_(True)

        for layer, weight in zip(self.target_layers, self.FUSION_WEIGHTS):
            with torch.enable_grad():
                with LayerCAM(model=self.model, target_layers=[layer]) as cam:
                    grayscale = cam(
                        input_tensor=input_tensor,
                        targets=targets,
                    )[0]  # (H_layer, W_layer)

            # Upsample to output_size
            layer_tensor = torch.from_numpy(grayscale).unsqueeze(0).unsqueeze(0)
            upsampled = F.interpolate(
                layer_tensor,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            fused += weight * upsampled

        # Min-max normalise to [0, 1]
        fused = self._normalise(fused)
        return fused

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    def generate_batch(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> list[np.ndarray]:
        """
        Generate heatmaps for a batch of images.

        Args:
            input_tensor: (B, 3, H, W) normalised float tensor.
            target_class: Class index for CAM.

        Returns:
            list of (H, W) float32 heatmaps, length B.
        """
        heatmaps = []
        for i in range(input_tensor.shape[0]):
            heatmaps.append(
                self.generate(input_tensor[i].unsqueeze(0), target_class=target_class)
            )
        return heatmaps
