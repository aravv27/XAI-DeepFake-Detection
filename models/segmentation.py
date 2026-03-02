"""
Semantic Segmentation Module using Pre-trained DeepLabV3+.

This module provides pixel-wise semantic segmentation using
the DeepLabV3+ model pre-trained on COCO dataset.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision import transforms
from typing import Optional, Tuple, Dict
from utils.labels import get_coco_segmentation_labels


class SemanticSegmenter:
    """
    Semantic segmentation using pre-trained DeepLabV3+.
    
    Provides pixel-wise class predictions for 21 COCO categories.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the segmenter with pre-trained DeepLabV3+.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load pre-trained DeepLabV3+ with ResNet101 backbone
        self.model = models.segmentation.deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load COCO segmentation labels
        self.labels = get_coco_segmentation_labels()
        self.num_classes = len(self.labels)
        
        # Define color palette for visualization
        self.color_palette = self._create_color_palette()
        
        print(f"SemanticSegmenter initialized on {self.device}")
    
    def _create_color_palette(self) -> np.ndarray:
        """Create a color palette for segmentation visualization."""
        # Use a predefined palette for COCO classes
        palette = np.array([
            [0, 0, 0],        # 0: background
            [128, 0, 0],      # 1: aeroplane
            [0, 128, 0],      # 2: bicycle
            [128, 128, 0],    # 3: bird
            [0, 0, 128],      # 4: boat
            [128, 0, 128],    # 5: bottle
            [0, 128, 128],    # 6: bus
            [128, 128, 128],  # 7: car
            [64, 0, 0],       # 8: cat
            [192, 0, 0],      # 9: chair
            [64, 128, 0],     # 10: cow
            [192, 128, 0],    # 11: dining table
            [64, 0, 128],     # 12: dog
            [192, 0, 128],    # 13: horse
            [64, 128, 128],   # 14: motorbike
            [192, 128, 128],  # 15: person
            [0, 64, 0],       # 16: potted plant
            [128, 64, 0],     # 17: sheep
            [0, 192, 0],      # 18: sofa
            [128, 192, 0],    # 19: train
            [0, 64, 128],     # 20: tv/monitor
        ], dtype=np.uint8)
        return palette
    
    def segment(self, image: Image.Image) -> np.ndarray:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            Per-pixel class predictions as numpy array (H, W)
        """
        # Store original size
        original_size = image.size  # (W, H)
        
        # Preprocess
        input_tensor = self.preprocess(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out']
        
        # Get class predictions
        predictions = torch.argmax(output, dim=1).squeeze(0)
        segment_map = predictions.cpu().numpy()
        
        # Resize back to original size
        segment_map = np.array(
            Image.fromarray(segment_map.astype(np.uint8)).resize(
                original_size, resample=Image.NEAREST
            )
        )
        
        return segment_map
    
    def get_class_mask(
        self, 
        segment_map: np.ndarray, 
        class_id: int
    ) -> np.ndarray:
        """
        Extract binary mask for a specific class.
        
        Args:
            segment_map: Full segmentation map (H, W)
            class_id: Class ID to extract
            
        Returns:
            Binary mask where class pixels are 1
        """
        return (segment_map == class_id).astype(np.float32)
    
    def get_colored_segmentation(
        self, 
        segment_map: np.ndarray
    ) -> np.ndarray:
        """
        Convert segmentation map to colored visualization.
        
        Args:
            segment_map: Per-pixel class predictions (H, W)
            
        Returns:
            RGB colored segmentation (H, W, 3)
        """
        colored = self.color_palette[segment_map]
        return colored
    
    def get_segments_info(
        self, 
        segment_map: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Get information about all segments in the image.
        
        Args:
            segment_map: Per-pixel class predictions (H, W)
            
        Returns:
            Dictionary mapping class_id to segment info (name, pixel_count, percentage)
        """
        total_pixels = segment_map.size
        unique_classes = np.unique(segment_map)
        
        segments_info = {}
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            
            mask = segment_map == class_id
            pixel_count = np.sum(mask)
            percentage = (pixel_count / total_pixels) * 100
            
            segments_info[int(class_id)] = {
                'name': self.labels[class_id],
                'pixel_count': int(pixel_count),
                'percentage': float(percentage),
                'mask': mask
            }
        
        return segments_info
    
    def get_label_name(self, class_id: int) -> str:
        """Get human-readable label for a class ID."""
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"Unknown ({class_id})"
