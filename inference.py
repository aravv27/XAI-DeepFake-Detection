"""
inference.py — v2
End-to-end deepfake inference + forensic explanation CLI.

Contract (from v2_implementation_plan.md §4.8):
  Steps:
    1. Resize 384×384, normalize, XceptionNet → logits, softmax
    2. LayerCAM on fake class → attention_map
    3. BiSeNet → segment_map
    4. NodeFeatureExtractor → (N, 10) features
    5. GAT forward → edge_importance
    6. GNNExplainer → causal edge masks
    7. Generate .txt + .json + .png reports

  Output files per image:
    {image}_analysis.png   — 4-panel dashboard
    {image}_explanation.txt — human-readable forensic report
    {image}_result.json    — structured data
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.xception import XceptionNetClassifier
from models.face_parser import FaceParser
from models.attention import LayerCAMGenerator
from models.gat_explainer import (
    GATExplainer,
    NodeFeatureExtractor,
    build_pyg_data,
    create_gat_batch,
)
from utils.explanation import explain_graph, generate_json_report, save_report
from utils.visualization import create_dashboard


class DeepfakeInference:
    """End-to-end deepfake inference with forensic explanations."""

    def __init__(
        self,
        checkpoint_path: str,
        bisenet_checkpoint: str,
        device: str = "auto",
    ):
        self.device = torch.device(
            device if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- Load models ---
        print(f"[Inference] Device: {self.device}")

        # XceptionNet
        self.xception = XceptionNetClassifier(num_classes=2, pretrained=False).to(self.device)

        # GAT
        self.gat = GATExplainer(input_dim=10).to(self.device)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "xception" in ckpt:
            self.xception.load_state_dict(ckpt["xception"])
            self.gat.load_state_dict(ckpt["gat"])
        else:
            # Phase 1 checkpoint (xception only)
            self.xception.load_state_dict(ckpt)
        
        self.xception.eval()
        self.gat.eval()

        # BiSeNet
        self.face_parser = FaceParser(
            checkpoint_path=bisenet_checkpoint, device=str(self.device),
        )

        # LayerCAM
        self.cam_gen = LayerCAMGenerator(self.xception, output_size=(384, 384))

        # Node features
        self.nfe = NodeFeatureExtractor()

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def analyze(
        self,
        image_path: str,
        output_dir: str = "results",
        show: bool = False,
    ) -> dict:
        """
        Run full analysis on a single image.

        Returns:
            JSON report dict.
        """
        # --- 1. Load image ---
        img_pil = Image.open(image_path).convert("RGB")
        np_img = np.array(img_pil.resize((384, 384)))
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # --- 2. XceptionNet classification ---
        with torch.no_grad():
            logits = self.xception(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred_class = int(logits.argmax(1).item())
        confidence = float(probs[pred_class].item())
        classification = "FAKE" if pred_class == 1 else "REAL"

        print(f"  Classification: {classification} ({confidence:.1%})")

        # --- 3. LayerCAM attention ---
        attention_map = self.cam_gen.generate(input_tensor, target_class=1)

        # --- 4. BiSeNet segmentation ---
        segment_map = self.face_parser.parse(np_img)

        # --- 5. Node features ---
        present_ids = [int(s) for s in np.unique(segment_map) if s != 0]
        features, valid_ids = self.nfe.extract(np_img, segment_map, attention_map, present_ids)

        # --- 6. Build graph + GAT ---
        edge_index_np, _ = self.face_parser.build_face_graph_edges(segment_map)

        # Re-index edges to valid_ids
        all_node_ids = sorted(set(np.unique(segment_map).tolist()) - {0})
        id_to_new = {sid: i for i, sid in enumerate(valid_ids)}
        old_to_new = {}
        for old_i, sid in enumerate(all_node_ids):
            if sid in id_to_new:
                old_to_new[old_i] = id_to_new[sid]

        new_src, new_dst = [], []
        for j in range(edge_index_np.shape[1]):
            s, d = int(edge_index_np[0, j]), int(edge_index_np[1, j])
            if s in old_to_new and d in old_to_new:
                new_src.append(old_to_new[s])
                new_dst.append(old_to_new[d])

        if new_src:
            filt_edge = np.array([new_src, new_dst], dtype=np.int64)
        else:
            filt_edge = np.zeros((2, 0), dtype=np.int64)

        # --- 7. GNNExplainer ---
        edge_mask = np.zeros(filt_edge.shape[1], dtype=np.float32)
        explainer_success = False

        if features.shape[0] >= 2 and filt_edge.shape[1] > 0:
            data = build_pyg_data(features, filt_edge, label=pred_class)
            data = data.to(self.device)

            explain_result = explain_graph(self.gat, data, target_class=pred_class)
            edge_mask = explain_result["edge_mask"]
            explainer_success = explain_result["success"]

        # --- 8. Generate reports ---
        seg_info = self.face_parser.get_segment_info(segment_map)

        json_report = generate_json_report(
            classification=classification,
            confidence=confidence,
            segment_info=seg_info,
            edge_index=filt_edge,
            edge_mask=edge_mask,
            segment_ids=valid_ids,
            explainer_success=explainer_success,
            image_path=image_path,
        )

        # Save
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path, txt_path = save_report(json_report, output_dir, image_name)
        print(f"  📄 JSON: {json_path}")
        print(f"  📄 Text: {txt_path}")

        # Dashboard
        dashboard_path = os.path.join(output_dir, f"{image_name}_analysis.png")
        create_dashboard(
            image=np_img,
            heatmap=attention_map,
            segment_map=segment_map,
            json_report=json_report,
            save_path=dashboard_path,
            show=show,
        )
        print(f"  🖼️  Dashboard: {dashboard_path}")

        return json_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="v2 Deepfake Inference — forensic analysis with GNNExplainer",
    )
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--bisenet-checkpoint", "-b", required=True, help="Path to BiSeNet weights")
    parser.add_argument("--output", "-o", default="results", help="Output directory")
    parser.add_argument("--device", "-d", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--no-viz", action="store_true", help="Skip matplotlib display")
    args = parser.parse_args()

    engine = DeepfakeInference(
        checkpoint_path=args.checkpoint,
        bisenet_checkpoint=args.bisenet_checkpoint,
        device=args.device,
    )
    engine.analyze(
        image_path=args.image,
        output_dir=args.output,
        show=not args.no_viz,
    )


if __name__ == "__main__":
    main()
