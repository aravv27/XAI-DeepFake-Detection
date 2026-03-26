"""
verify_pipeline.py — v2
End-to-end integration test. MUST pass before any training epoch runs.

Contract (from v2_implementation_plan.md §5.1):
  Run with a single real face image. Assert ALL of the following:
    1. BiSeNet produces a non-uniform segment map (≥3 distinct segment IDs)
    2. Node feature matrix has shape (N, 10) with no NaN and no all-zero rows
    3. GAT forward pass produces logits with shape (1, 2)
    4. A single backward pass produces GAT parameter gradients with norm > 1e-6
    5. Inference pipeline produces a .json file with key_relationships populated

  If ANY assertion fails → training does NOT start.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.xception import XceptionNetClassifier
from models.face_parser import FaceParser
from models.attention import LayerCAMGenerator
from models.gat_explainer import GATExplainer, NodeFeatureExtractor, build_pyg_data, create_gat_batch
from utils.explanation import explain_graph, generate_json_report


def _load_image(path: str, size: int = 384) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess a test image. Returns (tensor, numpy_rgb)."""
    img = Image.open(path).convert("RGB")
    np_img = np.array(img.resize((size, size)))

    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tfm(img).unsqueeze(0)  # (1, 3, 384, 384)
    return tensor, np_img


def verify(
    image_path: str,
    bisenet_checkpoint: str,
    device: str = "auto",
) -> bool:
    """
    Run all 5 integration checks.

    Returns True if all pass. Prints detailed results for each check.
    """
    dev = torch.device(
        device if device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n{'='*60}")
    print(f"  v2 INTEGRATION VERIFICATION")
    print(f"  Device: {dev}  |  Image: {image_path}")
    print(f"{'='*60}\n")

    all_passed = True

    # --- Load models ---
    print("[INIT] Loading XceptionNet...")
    xception = XceptionNetClassifier(num_classes=2, pretrained=True).to(dev).eval()

    print("[INIT] Loading BiSeNet face parser...")
    face_parser = FaceParser(checkpoint_path=bisenet_checkpoint, device=str(dev))

    print("[INIT] Initialising GAT explainer...")
    gat = GATExplainer(input_dim=10).to(dev)

    print("[INIT] Loading test image...")
    input_tensor, np_image = _load_image(image_path)
    input_tensor = input_tensor.to(dev)

    # =================================================================
    # CHECK 1: BiSeNet produces non-uniform segments (≥3 distinct IDs)
    # =================================================================
    print("\n[CHECK 1] BiSeNet segmentation...")
    segment_map = face_parser.parse(np_image)
    unique_ids = np.unique(segment_map)
    num_unique = len(unique_ids)

    if num_unique >= 3:
        print(f"  ✅ PASS — {num_unique} distinct segment IDs: {unique_ids.tolist()}")
    else:
        print(f"  ❌ FAIL — Only {num_unique} segment IDs: {unique_ids.tolist()}")
        print(f"           Expected ≥3. BiSeNet may have random/corrupted weights.")
        all_passed = False

    # =================================================================
    # CHECK 2: Node features shape (N, 10), no NaN, no all-zero rows
    # =================================================================
    print("\n[CHECK 2] Node feature extraction...")
    cam_gen = LayerCAMGenerator(xception, output_size=(384, 384))
    attention_map = cam_gen.generate(input_tensor, target_class=1)

    present_ids = [int(s) for s in unique_ids if s != 0]
    nfe = NodeFeatureExtractor()
    features, valid_ids = nfe.extract(np_image, segment_map, attention_map, present_ids)

    checks_2 = []
    if features.shape[1] == 10:
        checks_2.append(f"  ✅ Feature dim = {features.shape[1]}")
    else:
        checks_2.append(f"  ❌ Feature dim = {features.shape[1]}, expected 10")
        all_passed = False

    if not np.any(np.isnan(features)):
        checks_2.append(f"  ✅ No NaN values")
    else:
        checks_2.append(f"  ❌ NaN found in feature matrix!")
        all_passed = False

    zero_rows = np.all(features == 0, axis=1).sum()
    if zero_rows == 0:
        checks_2.append(f"  ✅ No all-zero rows ({features.shape[0]} nodes)")
    else:
        checks_2.append(f"  ❌ {zero_rows} all-zero rows found")
        all_passed = False

    for line in checks_2:
        print(line)

    # =================================================================
    # CHECK 3: GAT forward pass → logits shape (1, 2)
    # =================================================================
    print("\n[CHECK 3] GAT forward pass...")
    edge_index, _ = face_parser.build_face_graph_edges(segment_map)

    # Re-index edges to match valid_ids
    id_to_new_idx = {sid: i for i, sid in enumerate(valid_ids)}
    all_node_ids = sorted(set(np.unique(segment_map).tolist()) - {0})
    old_to_new = {}
    old_idx = 0
    for sid in all_node_ids:
        if sid in id_to_new_idx:
            old_to_new[old_idx] = id_to_new_idx[sid]
        old_idx += 1

    # Filter edges to only include valid nodes
    valid_edge_mask = []
    new_src, new_dst = [], []
    for i in range(edge_index.shape[1]):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        if s in old_to_new and d in old_to_new:
            new_src.append(old_to_new[s])
            new_dst.append(old_to_new[d])
    if new_src:
        filtered_edge_index = np.array([new_src, new_dst], dtype=np.int64)
    else:
        filtered_edge_index = np.zeros((2, 0), dtype=np.int64)

    data = build_pyg_data(features, filtered_edge_index, label=1)
    batch = create_gat_batch([data])
    batch = batch.to(dev)

    gat.eval()
    with torch.no_grad():
        logits = gat(batch.x, batch.edge_index, batch.batch)

    if logits.shape == (1, 2):
        print(f"  ✅ PASS — GAT logits shape: {tuple(logits.shape)}")
    else:
        print(f"  ❌ FAIL — GAT logits shape: {tuple(logits.shape)}, expected (1, 2)")
        all_passed = False

    # =================================================================
    # CHECK 4: GAT backward pass → grad norm > 1e-6
    # =================================================================
    print("\n[CHECK 4] GAT gradient check...")
    gat.train()
    gat.zero_grad()
    logits = gat(batch.x, batch.edge_index, batch.batch)
    loss = torch.nn.CrossEntropyLoss()(logits, batch.y)
    loss.backward()

    total_norm = 0.0
    for p in gat.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > 1e-6:
        print(f"  ✅ PASS — GAT grad norm: {total_norm:.6f}")
    else:
        print(f"  ❌ FAIL — GAT grad norm: {total_norm:.6f} (must be > 1e-6)")
        all_passed = False

    # =================================================================
    # CHECK 5: JSON report with key_relationships populated
    # =================================================================
    print("\n[CHECK 5] Report generation...")
    gat.eval()
    explain_result = explain_graph(gat, data.to(dev), target_class=1)

    seg_info = face_parser.get_segment_info(segment_map)
    json_report = generate_json_report(
        classification="FAKE",
        confidence=0.95,
        segment_info=seg_info,
        edge_index=filtered_edge_index,
        edge_mask=explain_result["edge_mask"],
        segment_ids=valid_ids,
        explainer_success=explain_result["success"],
        image_path=image_path,
    )

    has_relationships = len(json_report.get("key_relationships", [])) > 0
    if has_relationships:
        print(f"  ✅ PASS — {len(json_report['key_relationships'])} relationships in report")
    else:
        print(f"  ❌ FAIL — key_relationships is empty")
        all_passed = False

    # Save test report to temp dir 
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(json_report, f, indent=2)
        print(f"  📄 Test report saved: {f.name}")

    # =================================================================
    # Final result
    # =================================================================
    print(f"\n{'='*60}")
    if all_passed:
        print("  ✅ ALL 5 CHECKS PASSED — pipeline is ready for training")
    else:
        print("  ❌ SOME CHECKS FAILED — DO NOT proceed with training")
        print("  Fix the failures above before running train.py")
    print(f"{'='*60}\n")

    return all_passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="v2 Integration Verification — must pass before training",
    )
    parser.add_argument("--image", "-i", required=True, help="Path to a real face image")
    parser.add_argument("--bisenet-checkpoint", "-b", required=True, help="Path to BiSeNet pretrained weights")
    parser.add_argument("--device", "-d", default="auto", help="Device: cuda, cpu, or auto")
    args = parser.parse_args()

    passed = verify(args.image, args.bisenet_checkpoint, args.device)
    sys.exit(0 if passed else 1)
