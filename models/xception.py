"""
XceptionNet Classifier for Deepfake Detection

XceptionNet architecture adapted for binary deepfake detection.
Uses timm for pretrained weights and exports intermediate layers for LayerCAM.

Reference:
- Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions
- Rössler et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. XceptionNet will use fallback.")


class XceptionNetClassifier(nn.Module):
    """
    XceptionNet adapted for binary deepfake detection.
    
    Features:
    - Pretrained on ImageNet, fine-tuned on FF++
    - Modified final layer for binary classification
    - Exposes intermediate layers for LayerCAM
    - Optionally returns features for GAT
    
    Args:
        pretrained: Load ImageNet pretrained weights
        num_classes: Number of output classes (default: 2 for real/fake)
        dropout: Dropout probability before final FC
        return_features: If True, also return intermediate features
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.5,
        return_features: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.return_features = return_features
        
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for XceptionNet. "
                "Install with: pip install timm"
            )
        
        # Load pretrained Xception from timm
        self.backbone = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension from backbone
        self.feature_dim = 2048  # Xception output channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # Layer hooks for LayerCAM
        self.layer_outputs: Dict[str, torch.Tensor] = {}
        self.layer_gradients: Dict[str, torch.Tensor] = {}
        self._register_hooks()
        
        print(f"[XceptionNet] Loaded {'pretrained' if pretrained else 'random'} model")
        print(f"             Output classes: {num_classes}")
    
    def _register_hooks(self):
        """Register forward/backward hooks for intermediate layers."""
        # Target layers for LayerCAM (early, mid, late)
        self.target_layer_names = [
            'block3',   # Early features (textures)
            'block6',   # Mid features  
            'block12',  # Late features (semantic)
        ]
        
        # Map layer names to actual modules
        # Xception structure: conv1, conv2, block1-12, conv3, conv4
        self._hooked_layers = {}
        
        for name, module in self.backbone.named_modules():
            if name in ['block3', 'block6', 'block12']:
                self._hooked_layers[name] = module
                
                # Forward hook
                module.register_forward_hook(
                    self._create_forward_hook(name)
                )
                
                # Backward hook for gradients
                module.register_full_backward_hook(
                    self._create_backward_hook(name)
                )
    
    def _create_forward_hook(self, name: str):
        """Create forward hook for a named layer."""
        def hook(module, input, output):
            self.layer_outputs[name] = output
        return hook
    
    def _create_backward_hook(self, name: str):
        """Create backward hook for gradients."""
        def hook(module, grad_input, grad_output):
            self.layer_gradients[name] = grad_output[0]
        return hook
    
    def get_target_layers(self) -> List[nn.Module]:
        """Get target layers for CAM methods."""
        return [self._hooked_layers[name] for name in self.target_layer_names 
                if name in self._hooked_layers]
    
    def get_last_conv_layer(self) -> nn.Module:
        """Get the last convolutional layer for standard GradCAM."""
        # Return block12 as the last meaningful conv block
        if 'block12' in self._hooked_layers:
            return self._hooked_layers['block12']
        # Fallback to conv4
        return self.backbone.conv4
    
    def forward(
        self, 
        x: torch.Tensor,
        return_cam_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            return_cam_features: If True, return layer outputs for CAM
        
        Returns:
            If return_cam_features=False:
                logits: (B, num_classes)
            If return_cam_features=True:
                logits: (B, num_classes)
                features: Dict of layer outputs
        """
        # Clear previous layer outputs
        self.layer_outputs = {}
        self.layer_gradients = {}
        
        # Forward through backbone
        features = self.backbone(x)  # (B, 2048, H/32, W/32)
        
        # Global pooling
        pooled = self.global_pool(features)  # (B, 2048, 1, 1)
        pooled = pooled.flatten(1)  # (B, 2048)
        
        # Dropout and classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_classes)
        
        if return_cam_features:
            return logits, self.layer_outputs.copy()
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature vector before classification head.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            features: (B, 2048)
        """
        features = self.backbone(x)
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        return pooled
    
    def get_layer_features(
        self, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get intermediate layer features for GAT node construction.
        
        Returns dict with features from block3, block6, block12.
        """
        # Forward pass to populate layer_outputs
        _ = self.forward(x)
        return self.layer_outputs.copy()
    
    def freeze_backbone(self, freeze_until: str = 'block6'):
        """
        Freeze backbone layers up to specified block.
        
        Args:
            freeze_until: Freeze all layers up to and including this block
        """
        freeze = True
        for name, param in self.backbone.named_parameters():
            if freeze:
                param.requires_grad = False
            if freeze_until in name:
                freeze = False
        
        frozen_count = sum(1 for p in self.backbone.parameters() 
                         if not p.requires_grad)
        total_count = sum(1 for p in self.backbone.parameters())
        print(f"[XceptionNet] Frozen {frozen_count}/{total_count} backbone params")
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def load_xception_for_deepfake(
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda'
) -> XceptionNetClassifier:
    """
    Load XceptionNet for deepfake detection.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        checkpoint_path: Path to fine-tuned checkpoint (optional)
        device: Device to load model on
    
    Returns:
        Loaded XceptionNet model
    """
    model = XceptionNetClassifier(
        pretrained=pretrained,
        num_classes=2,
        dropout=0.5
    )
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


# Lightweight alternative for testing without timm
class SimpleXceptionBlock(nn.Module):
    """Simplified Xception-style separable conv block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, stride=stride, padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out + self.shortcut(x))
        return out


class LightweightXception(nn.Module):
    """
    Lightweight Xception-like model for testing.
    Use this if timm is not available.
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.block1 = SimpleXceptionBlock(64, 128, stride=2)
        self.block2 = SimpleXceptionBlock(128, 256, stride=2)
        self.block3 = SimpleXceptionBlock(256, 728, stride=2)
        
        # Middle flow
        self.middle = nn.Sequential(*[
            SimpleXceptionBlock(728, 728) for _ in range(4)
        ])
        
        # Exit flow
        self.block4 = SimpleXceptionBlock(728, 1024, stride=2)
        self.block5 = SimpleXceptionBlock(1024, 2048)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
