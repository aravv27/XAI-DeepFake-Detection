"""
Visualization Utilities for XAI Outputs.

Provides functions to create attention heatmap overlays,
segmentation visualizations, and comparison grids.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
import cv2


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay attention heatmap on original image.
    
    Args:
        image: Original RGB image (H, W, 3) with values in [0, 255]
        heatmap: Attention heatmap (H, W) with values in [0, 1]
        alpha: Opacity of the heatmap overlay
        colormap: Matplotlib colormap name
        
    Returns:
        RGB image with heatmap overlay (H, W, 3)
    """
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize heatmap to match image if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap to heatmap
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend images
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def overlay_segmentation(
    image: np.ndarray,
    segment_map: np.ndarray,
    color_palette: np.ndarray,
    alpha: float = 0.5,
    draw_boundaries: bool = True
) -> np.ndarray:
    """
    Overlay segmentation mask on original image.
    
    Args:
        image: Original RGB image (H, W, 3)
        segment_map: Per-pixel class predictions (H, W)
        color_palette: Color for each class (num_classes, 3)
        alpha: Opacity of the segmentation overlay
        draw_boundaries: Whether to draw segment boundaries
        
    Returns:
        RGB image with segmentation overlay
    """
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize segment_map to match image if needed
    if segment_map.shape[:2] != image.shape[:2]:
        segment_map = cv2.resize(
            segment_map.astype(np.uint8), 
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Create colored segmentation
    colored_seg = color_palette[segment_map]
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1 - alpha, colored_seg, alpha, 0)
    
    # Draw boundaries
    if draw_boundaries:
        # Find edges using Sobel
        edges = np.zeros_like(segment_map)
        for class_id in np.unique(segment_map):
            if class_id == 0:  # Skip background
                continue
            mask = (segment_map == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    
    return overlay


def create_comparison_grid(
    original: np.ndarray,
    heatmap_overlay: np.ndarray,
    segmentation_overlay: np.ndarray,
    combined_overlay: np.ndarray,
    titles: List[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> np.ndarray:
    """
    Create a 2x2 grid comparing different visualizations.
    
    Args:
        original: Original image
        heatmap_overlay: Image with heatmap overlay
        segmentation_overlay: Image with segmentation overlay
        combined_overlay: Image with both overlays
        titles: Titles for each subplot
        figsize: Figure size in inches
        
    Returns:
        Grid image as numpy array (H, W, 3)
    """
    if titles is None:
        titles = ['Original', 'Attention Heatmap', 'Segmentation', 'Combined']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    images = [original, heatmap_overlay, segmentation_overlay, combined_overlay]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    grid_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_array = grid_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return grid_array


def add_text_annotation(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Add text annotation to an image.
    
    Args:
        image: Input image (H, W, 3)
        text: Text to add
        position: (x, y) position for text
        font_scale: Font size scale
        color: Text color (RGB)
        background_color: Background rectangle color (RGB)
        
    Returns:
        Image with text annotation
    """
    image = image.copy()
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(
        image,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        background_color,
        -1
    )
    
    # Draw text
    cv2.putText(image, text, position, font, font_scale, color, 2, cv2.LINE_AA)
    
    return image


def create_legend(
    class_names: List[str],
    colors: List[Tuple[int, int, int]],
    figsize: Tuple[int, int] = (4, 6)
) -> np.ndarray:
    """
    Create a color legend for segmentation classes.
    
    Args:
        class_names: List of class names
        colors: RGB colors for each class
        figsize: Figure size
        
    Returns:
        Legend image as numpy array
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    patches = []
    for name, color in zip(class_names, colors):
        # Normalize color to [0, 1]
        normalized_color = tuple(c / 255.0 for c in color)
        patch = mpatches.Patch(color=normalized_color, label=name)
        patches.append(patch)
    
    ax.legend(handles=patches, loc='center', fontsize=10)
    ax.axis('off')
    
    fig.canvas.draw()
    legend_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    legend_array = legend_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return legend_array


def generate_explanation_text(
    class_name: str,
    confidence: float,
    coverage: float,
    iou: float,
    segment_label: str
) -> str:
    """
    Generate human-readable explanation text.
    
    Args:
        class_name: Predicted class name
        confidence: Prediction confidence (0-1)
        coverage: Attention coverage percentage (0-100)
        iou: Intersection over Union score (0-1)
        segment_label: Name of the segmented region
        
    Returns:
        Human-readable explanation string
    """
    explanation = (
        f"Classification: {class_name} ({confidence*100:.1f}% confidence)\n"
        f"Focus Analysis: Model attended primarily to the {segment_label} region "
        f"({coverage:.1f}% of total attention)\n"
        f"Alignment: {iou*100:.1f}% overlap between attention and segmented "
        f"{segment_label} regions\n"
        f"Interpretation: The model identified the subject by focusing on "
        f"distinctive {segment_label} features"
    )
    
    return explanation


def save_visualization(
    image: np.ndarray,
    output_path: str,
    dpi: int = 300
) -> None:
    """
    Save visualization to file.
    
    Args:
        image: Image to save (H, W, 3)
        output_path: Output file path
        dpi: Resolution for saved image
    """
    # Convert to PIL and save
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    pil_image.save(output_path, dpi=(dpi, dpi))
    print(f"Saved visualization to {output_path}")
