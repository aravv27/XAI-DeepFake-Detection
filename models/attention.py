"""
Attention Map Generation using Grad-CAM, Score-CAM, and LayerCAM.

This module provides attention heatmap generation for classification models
using the pytorch-grad-cam library. Extended with LayerCAM for multi-scale
attention fusion from XceptionNet for deepfake detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Union, List, Dict, Tuple
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.preprocessing import preprocess_for_classification


class AttentionGenerator:
    """
    Generate attention heatmaps using Grad-CAM variants.
    
    Supports:
    - GradCAM: Basic gradient-weighted activation mapping
    - GradCAM++: Improved localization for multiple instances
    - ScoreCAM: Gradient-free, more reliable localization
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        method: str = 'gradcam',
        target_layers: Optional[list] = None
    ):
        """
        Initialize attention generator.
        
        Args:
            model: Pre-trained classification model (e.g., ResNet50)
            method: Attention method ('gradcam', 'gradcam++', 'scorecam')
            target_layers: Layers to compute CAM from (default: last conv layer)
        """
        self.model = model
        self.method = method.lower()
        
        # For ResNet50, use layer4 (last conv block)
        if target_layers is None:
            target_layers = [model.layer4[-1]]
        
        self.target_layers = target_layers
        
        # Initialize CAM method
        if self.method == 'gradcam':
            self.cam = GradCAM(model=model, target_layers=target_layers)
        elif self.method == 'gradcam++':
            self.cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        elif self.method == 'scorecam':
            self.cam = ScoreCAM(model=model, target_layers=target_layers)
        elif self.method == 'layercam':
            self.cam = LayerCAM(model=model, target_layers=target_layers)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gradcam', 'gradcam++', 'scorecam', or 'layercam'")
        
        print(f"AttentionGenerator initialized with {method}")
    
    def generate_heatmap(
        self, 
        image: Image.Image,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate attention heatmap for an image.
        
        Args:
            image: PIL Image to analyze
            target_class: Class ID to generate attention for (None = predicted class)
            
        Returns:
            Normalized heatmap as numpy array (H, W) with values in [0, 1]
        """
        # Preprocess image
        input_tensor = preprocess_for_classification(image)
        
        # Set up target
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # Get the first (and only) image's CAM
        heatmap = grayscale_cam[0, :]
        
        return heatmap
    
    def generate_heatmap_from_tensor(
        self, 
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate attention heatmap from preprocessed tensor.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, 224, 224)
            target_class: Class ID to generate attention for (None = predicted class)
            
        Returns:
            Normalized heatmap as numpy array (H, W) with values in [0, 1]
        """
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0, :]
        
        return heatmap
    
    def generate_overlay(
        self, 
        image: Image.Image,
        target_class: Optional[int] = None,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Generate heatmap overlaid on original image.
        
        Args:
            image: PIL Image to analyze
            target_class: Class ID to generate attention for
            colormap: Matplotlib colormap name
            
        Returns:
            RGB image with heatmap overlay (H, W, 3) with values in [0, 1]
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(image, target_class)
        
        # Resize image to match heatmap if needed
        image_resized = image.resize((224, 224))
        rgb_img = np.array(image_resized) / 255.0
        
        # Create overlay
        overlay = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        
        return overlay
    
    def threshold_heatmap(
        self, 
        heatmap: np.ndarray, 
        threshold: float = 0.2
    ) -> np.ndarray:
        """
        Apply threshold to get binary attention mask.
        
        Args:
            heatmap: Attention heatmap (H, W)
            threshold: Threshold value (top X% of attention)
            
        Returns:
            Binary mask where high attention regions are 1
        """
        # Get threshold value (top percentage)
        threshold_value = np.percentile(heatmap, (1 - threshold) * 100)
        binary_mask = (heatmap >= threshold_value).astype(np.float32)
        
        return binary_mask


class LayerCAMGenerator:
    """
    Multi-scale LayerCAM for XceptionNet deepfake detection.
    
    Extracts attention from multiple XceptionNet blocks and fuses them
    to capture both texture (early) and semantic (late) features.
    
    LayerCAM is better than GradCAM for fine-grained localization,
    especially for detecting manipulation artifacts.
    
    Reference:
    - Jiang et al. (2021). LayerCAM: Exploring Hierarchical Class Activation Maps
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Optional[List[str]] = None,
        use_cuda: bool = True
    ):
        """
        Initialize multi-scale LayerCAM generator.
        
        Args:
            model: XceptionNet classifier with layer hooks
            target_layers: Layer names to extract CAM from 
                          (default: ['block3', 'block6', 'block12'])
            use_cuda: Use GPU if available
        """
        self.model = model
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Default layers for XceptionNet
        if target_layers is None:
            target_layers = ['block3', 'block6', 'block12']
        self.target_layer_names = target_layers
        
        # Get actual layer modules
        self.target_layers = self._get_target_layers()
        
        # Fusion weights (learnable or fixed)
        # Early layers detect texture, late layers detect semantics
        self.fusion_weights = [0.2, 0.3, 0.5]  # block3, block6, block12
        
        print(f"[LayerCAMGenerator] Initialized with layers: {target_layers}")
    
    def _get_target_layers(self) -> List[torch.nn.Module]:
        """Extract target layer modules from model."""
        layers = []
        
        # Try to get layers from XceptionNet classifier
        if hasattr(self.model, 'get_target_layers'):
            return self.model.get_target_layers()
        
        # Fallback: try to find layers by name
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in self.target_layer_names):
                layers.append(module)
        
        if not layers:
            # Last resort: use the last conv layer
            if hasattr(self.model, 'get_last_conv_layer'):
                layers = [self.model.get_last_conv_layer()]
            else:
                raise ValueError("Could not find target layers in model")
        
        return layers
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate fused multi-layer attention map.
        
        Args:
            input_tensor: Preprocessed image tensor (B, 3, H, W)
            target_class: Target class for CAM (None = use predicted)
            output_size: Output size (H, W), default = input size
        
        Returns:
            Fused attention map (H, W) normalized to [0, 1]
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        if output_size is None:
            output_size = (input_tensor.shape[2], input_tensor.shape[3])
        
        # Generate LayerCAM for each target layer
        layer_cams = []
        
        for layer, weight in zip(self.target_layers, self.fusion_weights):
            cam = LayerCAM(model=self.model, target_layers=[layer])
            
            targets = None
            if target_class is not None:
                targets = [ClassifierOutputTarget(target_class)]
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            cam_map = grayscale_cam[0]  # (H, W)
            
            # Resize to output size
            cam_resized = F.interpolate(
                torch.from_numpy(cam_map).unsqueeze(0).unsqueeze(0).float(),
                size=output_size,
                mode='bilinear',
                align_corners=False
            )[0, 0].numpy()
            
            layer_cams.append(cam_resized * weight)
        
        # Fuse layer CAMs
        fused_cam = np.sum(layer_cams, axis=0)
        
        # Normalize to [0, 1]
        fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)
        
        return fused_cam
    
    def generate_with_features(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        """
        Generate CAM and return intermediate features for GAT.
        
        Returns:
            cam: Fused attention map
            features: Dict of layer features
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        # Get CAM
        cam = self.generate(input_tensor, target_class)
        
        # Get layer features
        if hasattr(self.model, 'get_layer_features'):
            features = self.model.get_layer_features(input_tensor)
        else:
            features = {}
        
        return cam, features
    
    def visualize_multi_scale(
        self,
        image: np.ndarray,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate visualizations for each layer and fused.
        
        Returns dict with:
        - 'block3': Early layer CAM
        - 'block6': Mid layer CAM  
        - 'block12': Late layer CAM
        - 'fused': Fused multi-scale CAM
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        output_size = (image.shape[0], image.shape[1])
        
        visualizations = {}
        
        for layer, name in zip(self.target_layers, self.target_layer_names):
            cam = LayerCAM(model=self.model, target_layers=[layer])
            
            targets = None
            if target_class is not None:
                targets = [ClassifierOutputTarget(target_class)]
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            
            # Resize
            cam_resized = F.interpolate(
                torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0).float(),
                size=output_size,
                mode='bilinear',
                align_corners=False
            )[0, 0].numpy()
            
            # Create overlay
            rgb_img = image.astype(np.float32) / 255.0
            overlay = show_cam_on_image(rgb_img, cam_resized, use_rgb=True)
            visualizations[name] = overlay
        
        # Fused
        fused_cam = self.generate(input_tensor, target_class, output_size)
        rgb_img = image.astype(np.float32) / 255.0
        visualizations['fused'] = show_cam_on_image(rgb_img, fused_cam, use_rgb=True)
        
        return visualizations
