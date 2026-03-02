"""
XAI Classification + Segmentation Alignment System

Main CLI interface for running the complete XAI pipeline:
1. Image classification with ResNet50
2. Attention heatmap generation (Grad-CAM/Score-CAM)
3. Semantic segmentation with DeepLabV3+
4. Alignment metrics calculation
5. Visualization generation
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.classifier import ImageClassifier
from models.attention import AttentionGenerator
from models.segmentation import SemanticSegmenter
from utils.preprocessing import load_image, resize_to_match
from utils.alignment import calculate_all_metrics, threshold_attention
from utils.visualization import (
    overlay_heatmap,
    overlay_segmentation,
    create_comparison_grid,
    add_text_annotation,
    generate_explanation_text,
    save_visualization
)
from utils.labels import find_matching_coco_class, get_coco_segmentation_labels


def run_pipeline(
    image_path: str,
    output_dir: str,
    attention_method: str = 'gradcam',
    attention_threshold: float = 0.2,
    device: str = None
):
    """
    Run the complete XAI pipeline on an image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        attention_method: 'gradcam', 'gradcam++', or 'scorecam'
        attention_threshold: Threshold for attention mask (0.2 = top 20%)
        device: 'cuda', 'cpu', or None for auto
    """
    print("=" * 60)
    print("XAI Classification + Segmentation Alignment System")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"\n[1/6] Loading image: {image_path}")
    image = load_image(image_path)
    original_array = np.array(image)
    print(f"      Image size: {image.size}")
    
    # Initialize models
    print(f"\n[2/6] Loading classification model (ResNet50)...")
    classifier = ImageClassifier(device=device)
    
    print(f"\n[3/6] Running classification...")
    predictions = classifier.classify(image, top_k=5)
    
    top_class_id, top_class_name, top_confidence = predictions[0]
    print(f"      Top prediction: {top_class_name} ({top_confidence*100:.1f}%)")
    print("      Top 5 predictions:")
    for i, (cid, cname, conf) in enumerate(predictions):
        print(f"        {i+1}. {cname}: {conf*100:.1f}%")
    
    # Generate attention heatmap
    print(f"\n[4/6] Generating attention heatmap ({attention_method})...")
    attention_gen = AttentionGenerator(
        model=classifier.get_model(),
        method=attention_method
    )
    heatmap = attention_gen.generate_heatmap(image, target_class=top_class_id)
    print(f"      Heatmap shape: {heatmap.shape}")
    
    # Semantic segmentation
    print(f"\n[5/6] Running semantic segmentation (DeepLabV3+)...")
    segmenter = SemanticSegmenter(device=device)
    segment_map = segmenter.segment(image)
    segments_info = segmenter.get_segments_info(segment_map)
    
    print(f"      Found {len(segments_info)} segments:")
    for class_id, info in segments_info.items():
        print(f"        - {info['name']}: {info['percentage']:.1f}%")
    
    # Find matching segment for predicted class
    coco_labels = get_coco_segmentation_labels()
    matching_coco = find_matching_coco_class(top_class_name)
    matching_class_id = None
    
    if matching_coco != "unknown":
        for class_id, info in segments_info.items():
            if info['name'].lower() == matching_coco.lower():
                matching_class_id = class_id
                break
    
    # Calculate alignment metrics
    print(f"\n[6/6] Calculating alignment metrics...")
    
    # Resize heatmap to match segment map
    heatmap_resized = resize_to_match(
        Image.fromarray((heatmap * 255).astype(np.uint8)), 
        segment_map.shape[::-1]
    ) / 255.0
    
    if matching_class_id is not None:
        segment_mask = segments_info[matching_class_id]['mask'].astype(np.float32)
        metrics = calculate_all_metrics(heatmap_resized, segment_mask, attention_threshold)
        segment_label = segments_info[matching_class_id]['name']
        print(f"      Matching segment: {segment_label}")
    else:
        # Use the largest non-background segment
        if segments_info:
            largest_segment = max(segments_info.items(), key=lambda x: x[1]['pixel_count'])
            segment_mask = largest_segment[1]['mask'].astype(np.float32)
            metrics = calculate_all_metrics(heatmap_resized, segment_mask, attention_threshold)
            segment_label = largest_segment[1]['name']
            print(f"      Using largest segment: {segment_label}")
        else:
            segment_mask = np.zeros_like(segment_map).astype(np.float32)
            metrics = {'iou': 0, 'coverage': 0, 'precision': 0, 'recall': 0}
            segment_label = "unknown"
    
    print(f"      IoU: {metrics['iou']*100:.1f}%")
    print(f"      Coverage: {metrics['coverage']:.1f}%")
    print(f"      Precision: {metrics['precision']*100:.1f}%")
    print(f"      Recall: {metrics['recall']*100:.1f}%")
    
    # Generate visualizations
    print("\n" + "-" * 60)
    print("Generating visualizations...")
    
    # Resize heatmap to original image size
    heatmap_full = resize_to_match(
        Image.fromarray((heatmap * 255).astype(np.uint8)),
        (original_array.shape[1], original_array.shape[0])
    ) / 255.0
    
    # Heatmap overlay
    heatmap_overlay = overlay_heatmap(original_array, heatmap_full, alpha=0.4)
    
    # Segmentation overlay
    segment_overlay = overlay_segmentation(
        original_array,
        segment_map,
        segmenter.color_palette,
        alpha=0.4
    )
    
    # Combined overlay
    combined = overlay_heatmap(segment_overlay, heatmap_full, alpha=0.3)
    
    # Add prediction text
    text = f"{top_class_name}: {top_confidence*100:.0f}%"
    heatmap_overlay = add_text_annotation(heatmap_overlay, text)
    combined = add_text_annotation(combined, text)
    
    # Create comparison grid
    grid = create_comparison_grid(
        original_array,
        heatmap_overlay,
        segment_overlay,
        combined,
        titles=['Original', f'Attention ({attention_method})', 'Segmentation', 'Combined']
    )
    
    # Save outputs
    output_name = Path(image_path).stem
    
    save_visualization(heatmap_overlay, os.path.join(output_dir, f"{output_name}_heatmap.png"))
    save_visualization(segment_overlay, os.path.join(output_dir, f"{output_name}_segmentation.png"))
    save_visualization(combined, os.path.join(output_dir, f"{output_name}_combined.png"))
    save_visualization(grid, os.path.join(output_dir, f"{output_name}_grid.png"))
    
    # Generate explanation
    explanation = generate_explanation_text(
        class_name=top_class_name,
        confidence=top_confidence,
        coverage=metrics['coverage'],
        iou=metrics['iou'],
        segment_label=segment_label
    )
    
    explanation_path = os.path.join(output_dir, f"{output_name}_explanation.txt")
    with open(explanation_path, 'w') as f:
        f.write(explanation)
    print(f"Saved explanation to {explanation_path}")
    
    # Save metrics JSON
    metrics_output = {
        "predicted_class": top_class_name,
        "predicted_class_id": int(top_class_id),
        "confidence": float(top_confidence),
        "attention_method": attention_method,
        "top_segment_label": segment_label,
        "iou_score": float(metrics['iou']),
        "coverage_percentage": float(metrics['coverage']),
        "precision": float(metrics['precision']),
        "recall": float(metrics['recall']),
        "attention_threshold": attention_threshold,
        "top_5_predictions": [
            {"class_id": int(cid), "class_name": cname, "confidence": float(conf)}
            for cid, cname, conf in predictions
        ]
    }
    
    metrics_path = os.path.join(output_dir, f"{output_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Print final explanation
    print("\n" + "=" * 60)
    print("EXPLANATION")
    print("=" * 60)
    print(explanation)
    print("=" * 60)
    
    print(f"\nAll outputs saved to: {output_dir}")
    
    return metrics_output


def main():
    parser = argparse.ArgumentParser(
        description="XAI Classification + Segmentation Alignment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image examples/sample_images/dog.jpg --output output/
  python main.py --image photo.png --method scorecam --threshold 0.3
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['gradcam', 'gradcam++', 'scorecam'],
        default='gradcam',
        help='Attention method (default: gradcam)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.2,
        help='Attention threshold (default: 0.2 = top 20%%)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Run pipeline
    run_pipeline(
        image_path=args.image,
        output_dir=args.output,
        attention_method=args.method,
        attention_threshold=args.threshold,
        device=args.device
    )


if __name__ == "__main__":
    main()
