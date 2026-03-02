"""
Inference Pipeline for Graph-Guided Deepfake Detection

Run inference on images and generate explanations with visualizations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.xception import XceptionNetClassifier
from models.face_parser import FaceParser, FACE_SEGMENTS, build_face_graph_edges
from models.gat_explainer import GATExplainer, NodeFeatureExtractor
from models.attention import LayerCAMGenerator
from utils.explanation import generate_explanation, generate_json_explanation
from utils.preprocessing import load_image
from utils.visualization import save_visualization


class DeepfakeInference:
    """
    Inference pipeline for deepfake detection with explainability.
    
    Combines XceptionNet, LayerCAM, Face Parsing, and GAT to:
    1. Classify image as real/fake
    2. Generate attention heatmap
    3. Parse facial segments  
    4. Compute suspicious segment relationships
    5. Generate human-readable explanations
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[DeepfakeInference] Using device: {self.device}")
        
        # Initialize models
        self._load_models(checkpoint_path)
        
        # Categories
        self.classes = ['REAL', 'FAKE']
    
    def _load_models(self, checkpoint_path: Optional[str]):
        """Load all required models."""
        print("Loading models...")
        
        # XceptionNet classifier
        self.xception = XceptionNetClassifier(
            pretrained=True,
            num_classes=2
        ).to(self.device)
        self.xception.eval()
        
        # Face parser
        self.face_parser = FaceParser(device=str(self.device))
        
        # LayerCAM
        self.layercam = LayerCAMGenerator(
            model=self.xception,
            use_cuda=self.device.type == 'cuda'
        )
        
        # Node feature extractor
        self.node_extractor = NodeFeatureExtractor(
            cnn_feature_dim=256,
            texture_feature_dim=2
        )
        
        # GAT
        self.gat = GATExplainer(
            in_features=self.node_extractor.feature_dim,
            hidden=128,
            heads=4
        ).to(self.device)
        self.gat.eval()
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.xception.load_state_dict(checkpoint['xception_state_dict'])
            if 'gat_state_dict' in checkpoint:
                self.gat.load_state_dict(checkpoint['gat_state_dict'])
            print("Checkpoint loaded successfully")
        else:
            print("No checkpoint - using pretrained weights")
        
        print("All models loaded")
    
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image | np.ndarray | str,
        return_details: bool = True
    ) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image, numpy array, or path to image
            return_details: If True, return full analysis details
        
        Returns:
            Dict with:
            - prediction: 'REAL' or 'FAKE'
            - confidence: Confidence score (0-1)
            - probabilities: [real_prob, fake_prob]
            - attention_map: (H, W) attention heatmap
            - segments: Face segment information
            - edge_importance: Segment relationship importance
            - explanation: Human-readable explanation text
        """
        # Load image
        if isinstance(image, str):
            image = load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get image as numpy array
        image_np = np.array(image)
        original_size = image.size  # (W, H)
        
        # Preprocess for classification (384x384 for Xception)
        transform = self._get_transform()
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 1. Classification
        logits = self.xception(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = int(logits.argmax(dim=1)[0].item())
        confidence = float(probs[pred_class])
        prediction = self.classes[pred_class]
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'real': float(probs[0]),
                'fake': float(probs[1])
            }
        }
        
        if not return_details:
            return result
        
        # 2. Attention map
        attention_map = self.layercam.generate(
            input_tensor,
            target_class=1,  # Always show attention for "fake" class
            output_size=(original_size[1], original_size[0])  # (H, W)
        )
        result['attention_map'] = attention_map
        
        # 3. Face parsing
        segment_map = self.face_parser.parse(image_np)
        segments_info = self.face_parser.get_segment_info(segment_map)
        result['segment_map'] = segment_map
        result['segments'] = segments_info
        
        # 4. Build graph and compute edge importance
        edge_index, present_segments = build_face_graph_edges(
            segment_map, fully_connected=True
        )
        
        segment_names = [FACE_SEGMENTS[i] for i in present_segments]
        
        if len(present_segments) >= 2:
            # Extract node features
            node_features = self._extract_node_features(
                attention_map, segment_map, present_segments, image_np
            )
            
            # GAT forward
            edge_index_gpu = edge_index.to(self.device)
            node_features_gpu = node_features.to(self.device)
            
            gat_output = self.gat(
                node_features_gpu,
                edge_index_gpu,
                return_attention=True
            )
            
            edge_importance = gat_output.get('edge_importance', None)
            
            if edge_importance is not None:
                edge_importance_dict = {}
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[:, i].numpy()
                    if src < len(segment_names) and dst < len(segment_names):
                        key = (segment_names[src], segment_names[dst])
                        edge_importance_dict[key] = float(edge_importance[i].cpu())
                result['edge_importance'] = edge_importance_dict
            else:
                result['edge_importance'] = {}
        else:
            result['edge_importance'] = {}
        
        # 5. Generate explanation
        result['explanation'] = generate_explanation(
            prediction=prediction,
            confidence=confidence,
            edge_importance=result['edge_importance'],
            segment_info=segments_info
        )
        
        result['explanation_json'] = generate_json_explanation(
            prediction=prediction,
            confidence=confidence,
            edge_importance=result['edge_importance'],
            segment_info=segments_info
        )
        
        return result
    
    def _get_transform(self):
        """Get image transform for inference."""
        import torchvision.transforms as T
        return T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _extract_node_features(
        self,
        attention_map: np.ndarray,
        segment_map: np.ndarray,
        segment_ids: List[int],
        image_np: np.ndarray
    ) -> torch.Tensor:
        """Extract node features for each segment."""
        features = []
        
        for seg_id in segment_ids:
            mask = (segment_map == seg_id).astype(np.float32)
            feat = np.zeros(self.node_extractor.feature_dim)
            
            if mask.sum() > 0:
                # Attention statistics
                attn_masked = attention_map * mask
                feat[0] = attn_masked.sum() / mask.sum()  # Mean attention
                feat[1] = attn_masked.std()  # Std attention
                feat[2] = attn_masked.max()  # Max attention
                feat[3] = mask.sum() / mask.size  # Area ratio
            
            features.append(feat)
        
        return torch.tensor(np.array(features), dtype=torch.float32)
    
    def visualize(
        self,
        result: Dict,
        image: Image.Image | np.ndarray,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> np.ndarray:
        """
        Create visualization of detection result.
        
        Args:
            result: Output from predict()
            image: Original image
            output_path: Path to save visualization
            show: If True, display the visualization
        
        Returns:
            Visualization as numpy array
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Attention heatmap
        if 'attention_map' in result:
            axes[0, 1].imshow(image_np)
            im = axes[0, 1].imshow(
                result['attention_map'],
                cmap='jet',
                alpha=0.5
            )
            axes[0, 1].set_title('Attention Map (LayerCAM)')
            axes[0, 1].axis('off')
            plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        # Face segments
        if 'segment_map' in result:
            segment_vis = self.face_parser.visualize(result['segment_map'])
            axes[1, 0].imshow(segment_vis)
            axes[1, 0].set_title('Face Segments')
            axes[1, 0].axis('off')
        
        # Classification result and top relationships
        axes[1, 1].axis('off')
        
        pred = result['prediction']
        conf = result['confidence']
        color = 'red' if pred == 'FAKE' else 'green'
        
        text_lines = [
            f"Classification: {pred}",
            f"Confidence: {conf*100:.1f}%\n"
        ]
        
        if result.get('edge_importance'):
            text_lines.append("Top Suspicious Relationships:")
            sorted_edges = sorted(
                result['edge_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for (seg1, seg2), imp in sorted_edges:
                status = "⚠️" if imp > 0.6 else "✓"
                text_lines.append(f"  {status} {seg1} ↔ {seg2}: {imp*100:.0f}%")
        
        axes[1, 1].text(
            0.1, 0.9,
            '\n'.join(text_lines),
            transform=axes[1, 1].transAxes,
            fontsize=12,
            verticalalignment='top',
            fontfamily='monospace',
            color=color
        )
        axes[1, 1].set_title('Analysis Results')
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        # Convert to array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return vis_array


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--device', '-d', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization display')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = DeepfakeInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Load image
    image = load_image(args.image)
    
    # Run inference
    print("\nRunning inference...")
    result = inference.predict(image, return_details=True)
    
    # Print explanation
    print("\n" + result['explanation'])
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save visualization
    image_name = Path(args.image).stem
    vis_path = os.path.join(args.output, f"{image_name}_analysis.png")
    inference.visualize(
        result, image,
        output_path=vis_path,
        show=not args.no_viz
    )
    
    # Save JSON results
    json_path = os.path.join(args.output, f"{image_name}_result.json")
    with open(json_path, 'w') as f:
        json.dump(result['explanation_json'], f, indent=2)
    print(f"Saved JSON results to {json_path}")
    
    # Save explanation text
    txt_path = os.path.join(args.output, f"{image_name}_explanation.txt")
    with open(txt_path, 'w') as f:
        f.write(result['explanation'])
    print(f"Saved explanation to {txt_path}")
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
