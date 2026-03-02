"""
Alignment Metrics for Attention-Segmentation Comparison.

Provides functions to calculate how well attention maps align
with semantic segmentation masks.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def calculate_iou(
    attention_mask: np.ndarray,
    segment_mask: np.ndarray
) -> float:
    """
    Calculate Intersection over Union between attention and segmentation masks.
    
    Args:
        attention_mask: Binary attention mask (H, W)
        segment_mask: Binary segmentation mask (H, W)
        
    Returns:
        IoU score between 0 and 1
    """
    # Ensure same shape
    if attention_mask.shape != segment_mask.shape:
        raise ValueError(f"Shape mismatch: {attention_mask.shape} vs {segment_mask.shape}")
    
    # Convert to binary if needed
    attention_binary = (attention_mask > 0.5).astype(np.float32)
    segment_binary = (segment_mask > 0.5).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.sum(attention_binary * segment_binary)
    union = np.sum(attention_binary) + np.sum(segment_binary) - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    iou = intersection / union
    
    return float(iou)


def calculate_coverage(
    attention_map: np.ndarray,
    segment_mask: np.ndarray
) -> float:
    """
    Calculate what percentage of attention falls within a segment.
    
    Coverage = sum(attention * segment) / sum(attention)
    
    Args:
        attention_map: Continuous attention values (H, W) in [0, 1]
        segment_mask: Binary segmentation mask (H, W)
        
    Returns:
        Coverage percentage between 0 and 100
    """
    # Ensure same shape
    if attention_map.shape != segment_mask.shape:
        raise ValueError(f"Shape mismatch: {attention_map.shape} vs {segment_mask.shape}")
    
    # Convert segment to binary
    segment_binary = (segment_mask > 0.5).astype(np.float32)
    
    # Calculate coverage
    attention_sum = np.sum(attention_map)
    
    if attention_sum == 0:
        return 0.0
    
    covered_attention = np.sum(attention_map * segment_binary)
    coverage = (covered_attention / attention_sum) * 100
    
    return float(coverage)


def calculate_precision_recall(
    attention_mask: np.ndarray,
    segment_mask: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate precision and recall of attention with respect to segmentation.
    
    Precision = TP / (TP + FP) - How much of the attention is correct
    Recall = TP / (TP + FN) - How much of the segment is attended to
    
    Args:
        attention_mask: Binary attention mask (H, W)
        segment_mask: Binary segmentation mask (H, W)
        
    Returns:
        Tuple of (precision, recall) between 0 and 1
    """
    # Convert to binary
    att = (attention_mask > 0.5).astype(np.float32)
    seg = (segment_mask > 0.5).astype(np.float32)
    
    # True positives: attention AND segment
    tp = np.sum(att * seg)
    
    # False positives: attention AND NOT segment
    fp = np.sum(att * (1 - seg))
    
    # False negatives: NOT attention AND segment
    fn = np.sum((1 - att) * seg)
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return float(precision), float(recall)


def threshold_attention(
    attention_map: np.ndarray,
    threshold: float = 0.2
) -> np.ndarray:
    """
    Apply threshold to get top percentage of attention.
    
    Args:
        attention_map: Continuous attention values (H, W) in [0, 1]
        threshold: Top percentage to keep (0.2 = top 20%)
        
    Returns:
        Binary mask where top attention regions are 1
    """
    # Get threshold value at the specified percentile
    threshold_value = np.percentile(attention_map, (1 - threshold) * 100)
    
    # Create binary mask
    binary_mask = (attention_map >= threshold_value).astype(np.float32)
    
    return binary_mask


def calculate_all_metrics(
    attention_map: np.ndarray,
    segment_mask: np.ndarray,
    threshold: float = 0.2
) -> Dict[str, float]:
    """
    Calculate all alignment metrics at once.
    
    Args:
        attention_map: Continuous attention values (H, W) in [0, 1]
        segment_mask: Binary segmentation mask (H, W)
        threshold: Threshold for binary attention mask
        
    Returns:
        Dictionary containing all metrics
    """
    # Threshold attention for IoU calculation
    attention_binary = threshold_attention(attention_map, threshold)
    
    # Calculate all metrics
    iou = calculate_iou(attention_binary, segment_mask)
    coverage = calculate_coverage(attention_map, segment_mask)
    precision, recall = calculate_precision_recall(attention_binary, segment_mask)
    
    return {
        'iou': iou,
        'coverage': coverage,
        'precision': precision,
        'recall': recall,
        'threshold': threshold
    }


def resize_mask(
    mask: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resize a mask to target shape using nearest neighbor interpolation.
    
    Args:
        mask: Input mask (H, W)
        target_shape: Target (H, W) tuple
        
    Returns:
        Resized mask
    """
    from PIL import Image
    
    # Convert to PIL Image
    mask_img = Image.fromarray(mask.astype(np.uint8))
    
    # Resize (PIL uses W, H order)
    resized = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    
    return np.array(resized).astype(np.float32)
