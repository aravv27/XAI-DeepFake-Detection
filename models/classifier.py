"""
Image Classification Module using Pre-trained ResNet50.

This module provides a wrapper around the pre-trained ResNet50 model
for image classification with ImageNet classes.
"""

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image
from typing import List, Tuple, Optional
from utils.preprocessing import preprocess_for_classification
from utils.labels import get_imagenet_labels


class ImageClassifier:
    """
    Image classifier using pre-trained ResNet50.
    
    Attributes:
        model: Pre-trained ResNet50 model
        device: torch device (cuda or cpu)
        labels: ImageNet class labels
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the classifier with pre-trained ResNet50.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load ImageNet labels
        self.labels = get_imagenet_labels()
        
        print(f"ImageClassifier initialized on {self.device}")
    
    def get_model(self) -> torch.nn.Module:
        """Return the underlying model for Grad-CAM access."""
        return self.model
    
    def classify(
        self, 
        image: Image.Image, 
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Classify an image and return top-k predictions.
        
        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples: (class_id, class_name, confidence)
        """
        # Preprocess image
        input_tensor = preprocess_for_classification(image)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        results = []
        for i in range(top_k):
            class_id = top_indices[0, i].item()
            class_name = self.labels[class_id]
            confidence = top_probs[0, i].item()
            results.append((class_id, class_name, confidence))
        
        return results
    
    def classify_with_tensor(
        self, 
        input_tensor: torch.Tensor, 
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Classify from a preprocessed tensor.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, 224, 224)
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples: (class_id, class_name, confidence)
        """
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
        
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        results = []
        for i in range(top_k):
            class_id = top_indices[0, i].item()
            class_name = self.labels[class_id]
            confidence = top_probs[0, i].item()
            results.append((class_id, class_name, confidence))
        
        return results
