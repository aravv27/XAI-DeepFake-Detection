"""
Image Preprocessing Utilities.

Provides preprocessing pipelines for classification and segmentation models.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Union


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image in RGB format
    """
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def preprocess_for_classification(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Preprocess image for classification models.
    
    Args:
        image: PIL Image to preprocess
        target_size: Target size (H, W)) for the model
        
    Returns:
        Preprocessed tensor (1, 3, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor


def preprocess_for_segmentation(
    image: Image.Image
) -> torch.Tensor:
    """
    Preprocess image for segmentation models.
    
    Note: Segmentation models can handle variable input sizes,
    so we don't resize the image.
    
    Args:
        image: PIL Image to preprocess
        
    Returns:
        Preprocessed tensor (1, 3, H, W)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    
    return tensor


def denormalize(
    tensor: torch.Tensor,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> torch.Tensor:
    """
    Reverse ImageNet normalization.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)
    
    return denorm


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array for visualization.
    
    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)
        
    Returns:
        Numpy array (H, W, C) with values in [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Move channels to last dimension
    array = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Scale to [0, 255]
    array = (array * 255).astype(np.uint8)
    
    return array


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array (H, W, C) with values in [0, 255]
        
    Returns:
        PIL Image
    """
    return Image.fromarray(array.astype(np.uint8))


def resize_to_match(
    image: Union[np.ndarray, Image.Image],
    target: Union[np.ndarray, Image.Image, Tuple[int, int]]
) -> np.ndarray:
    """
    Resize image to match target dimensions.
    
    Args:
        image: Image to resize
        target: Target image or (W, H) tuple
        
    Returns:
        Resized numpy array
    """
    # Get target size
    if isinstance(target, tuple):
        target_size = target
    elif isinstance(target, np.ndarray):
        target_size = (target.shape[1], target.shape[0])  # (W, H)
    else:
        target_size = target.size  # PIL Image size is (W, H)
    
    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize
    resized = image.resize(target_size, Image.BILINEAR)
    
    return np.array(resized)
