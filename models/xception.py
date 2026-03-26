"""
models/xception.py — v2
XceptionNet classifier with LayerCAM hooks.
Backbone: timm xception, pretrained on ImageNet.

Contract (from v2_implementation_plan.md §4.2):
  - Input:  384x384x3
  - Output: 2-class logits (real=0, fake=1)
  - Feature dim: 2048 after global avg pool
  - Dropout: 0.5 before final linear
  - CAM hook layers: block3 (textures), block6 (patterns), block12 (semantics)
  - Normalisation: ImageNet mean/std
  - NO fallback/lightweight class. If timm is missing → ImportError.
"""

import torch
import torch.nn as nn
import timm


class XceptionNetClassifier(nn.Module):
    """Binary deepfake classifier built on timm's Xception."""

    # ImageNet normalisation constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # --- backbone ---
        self.backbone = timm.create_model("xception", pretrained=pretrained)

        # Remove the original classifier head
        self.feature_dim = self.backbone.num_features  # 2048
        self.backbone.fc = nn.Identity()

        # --- custom classifier head ---
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # --- hook storage ---
        self._layer_outputs: dict[str, torch.Tensor] = {}
        self._hooks: list = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration for LayerCAM
    # ------------------------------------------------------------------

    def _get_hook_targets(self) -> dict[str, nn.Module]:
        """Map hook names to actual backbone sub-modules.

        timm's Xception layout:
          block1 … block12  (Sequential children)
          bn3, bn4, act3, act4, conv3, conv4  (exit flow)
        We hook block3, block6, block12 as defined in the v2 plan.
        """
        targets = {}
        for name in ("block3", "block6", "block12"):
            module = getattr(self.backbone, name, None)
            if module is not None:
                targets[name] = module
            else:
                raise AttributeError(
                    f"timm xception backbone has no attribute '{name}'. "
                    f"Available: {[n for n, _ in self.backbone.named_children()]}"
                )
        return targets

    def _register_hooks(self) -> None:
        """Attach forward hooks to store intermediate activations."""
        for name, module in self._get_hook_targets().items():
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(_module, _input, output):
            self._layer_outputs[name] = output
        return hook_fn

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_cam_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: (B, 3, 384, 384) normalised input.
            return_cam_features: if True, also return hooked layer outputs.

        Returns:
            logits (B, 2),  or  (logits, layer_outputs_dict).
        """
        self._layer_outputs.clear()

        features = self.backbone(x)           # (B, 2048)
        logits = self.classifier(self.dropout(features))  # (B, 2)

        if return_cam_features:
            return logits, dict(self._layer_outputs)  # shallow copy
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 2048) pooled feature vector (no classifier head)."""
        return self.backbone(x)

    def get_layer_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run forward and return hooked intermediate features."""
        self._layer_outputs.clear()
        self.backbone(x)
        return dict(self._layer_outputs)

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    _FREEZE_ORDER = [
        "conv1", "bn1", "act1",
        "conv2", "bn2", "act2",
        "block1", "block2", "block3",
        "block4", "block5", "block6",
        "block7", "block8", "block9",
        "block10", "block11", "block12",
        "conv3", "bn3", "act3",
        "conv4", "bn4", "act4",
    ]

    def freeze_backbone(self, freeze_until: str = "block6") -> None:
        """Freeze all backbone parameters up to and including *freeze_until*."""
        if freeze_until not in self._FREEZE_ORDER:
            raise ValueError(
                f"Unknown layer '{freeze_until}'. "
                f"Must be one of {self._FREEZE_ORDER}"
            )
        stop = self._FREEZE_ORDER.index(freeze_until)
        for name in self._FREEZE_ORDER[: stop + 1]:
            module = getattr(self.backbone, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter in the entire model."""
        for p in self.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # CAM helpers
    # ------------------------------------------------------------------

    def get_target_layers(self) -> list[nn.Module]:
        """Return the list of modules for pytorch-grad-cam."""
        targets = self._get_hook_targets()
        return [targets["block3"], targets["block6"], targets["block12"]]

    def get_last_conv_layer(self) -> nn.Module:
        """Return the deepest spatial feature layer (block12)."""
        return self._get_hook_targets()["block12"]
