"""
utils/explanation.py — v2
Edge attribution via GNNExplainer + GAT attention weights, plus report generation.

Contract (from v2_implementation_plan.md §4.8):
  - Primary: GNNExplainer (PyG built-in) for causal edge masks
  - Fallback: GAT's own attention weights (still meaningful, but not causally grounded)
  - Output formats: .txt forensic report, .json structured data
  - Template fallback: NONE. Reports show actual learned edge importance.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from torch_geometric.data import Data

from models.face_parser import SEGMENT_NAMES


# ---------------------------------------------------------------------------
# Edge importance extraction
# ---------------------------------------------------------------------------

def explain_graph(
    model: torch.nn.Module,
    data: Data,
    target_class: int = 1,
    epochs: int = 200,
    lr: float = 0.01,
) -> dict:
    """
    Get edge importance scores for a single graph.

    Strategy:
      1. Try PyG GNNExplainer for causal edge masks
      2. If that fails, use GAT's own attention weights (return_attention=True)
      3. If both fail, return zeros with success=False

    Args:
        model:        GATExplainer model (eval mode).
        data:         PyG Data with x, edge_index, y.
        target_class: Class to explain (1 = fake).
        epochs:       GNNExplainer optimisation epochs.
        lr:           GNNExplainer learning rate.

    Returns:
        Dict with 'edge_mask' (numpy), 'node_mask' (numpy),
        'success' (bool), 'method' (str).
    """
    device = next(model.parameters()).device
    data = data.to(device)

    # --- Attempt 1: PyG GNNExplainer ---
    try:
        from torch_geometric.explain import Explainer, GNNExplainer as PyGGNNExplainer

        explainer = Explainer(
            model=model,
            algorithm=PyGGNNExplainer(epochs=epochs, lr=lr),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
        )

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=torch.tensor([target_class], device=device),
            batch=torch.zeros(data.x.size(0), dtype=torch.long, device=device),
        )

        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        node_mask = explanation.node_mask.detach().cpu().numpy()

        # Check if the mask is actually meaningful (not all zeros)
        if edge_mask.max() > 1e-6:
            return {
                "edge_mask": edge_mask,
                "node_mask": node_mask,
                "success": True,
                "method": "GNNExplainer",
            }
    except Exception:
        pass  # Fall through to attention-based method

    # --- Attempt 2: GAT attention weights ---
    try:
        model.eval()
        with torch.no_grad():
            batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            logits, (attn1, attn2) = model(
                data.x, data.edge_index, batch=batch_vec, return_attention=True,
            )

        # attn1 = (edge_index_with_self_loops, attention_weights)
        # attn2 = (edge_index_with_self_loops, attention_weights)
        # Use layer 2 attention (closer to output, more meaningful)
        attn_edge_index, attn_weights = attn2

        # Map attention weights back to original edges
        num_edges = data.edge_index.size(1)
        edge_mask = np.zeros(num_edges, dtype=np.float32)

        # Build lookup from (src, dst) → attention weight
        attn_edge_np = attn_edge_index.cpu().numpy()
        attn_w_np = attn_weights.detach().cpu().numpy().flatten()
        attn_dict = {}
        for i in range(attn_edge_np.shape[1]):
            key = (int(attn_edge_np[0, i]), int(attn_edge_np[1, i]))
            attn_dict[key] = float(attn_w_np[i])

        # Fill edge_mask for original edges
        orig_edges = data.edge_index.cpu().numpy()
        for i in range(num_edges):
            key = (int(orig_edges[0, i]), int(orig_edges[1, i]))
            edge_mask[i] = attn_dict.get(key, 0.0)

        # Normalise to [0, 1]
        if edge_mask.max() > 0:
            edge_mask = edge_mask / edge_mask.max()

        node_mask = np.zeros((data.x.size(0), data.x.size(1)), dtype=np.float32)

        return {
            "edge_mask": edge_mask,
            "node_mask": node_mask,
            "success": True,
            "method": "GAT_attention",
        }

    except Exception as e:
        # Both methods failed
        return {
            "edge_mask": np.zeros(data.edge_index.shape[1], dtype=np.float32),
            "node_mask": np.zeros((data.x.shape[0], data.x.shape[1]), dtype=np.float32),
            "success": False,
            "method": "none",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_relationships(
    edge_index: np.ndarray,
    edge_mask: np.ndarray,
    segment_ids: list[int],
    explainer_success: bool,
) -> list[dict]:
    """Build sorted list of edge relationships with importance scores."""
    relationships = []
    num_edges = edge_index.shape[1]

    for i in range(num_edges):
        src_idx, dst_idx = int(edge_index[0, i]), int(edge_index[1, i])
        # Skip reverse edges (only report each pair once)
        if src_idx >= dst_idx:
            continue

        src_id = segment_ids[src_idx] if src_idx < len(segment_ids) else -1
        dst_id = segment_ids[dst_idx] if dst_idx < len(segment_ids) else -1

        score = float(edge_mask[i]) if i < len(edge_mask) else 0.0

        relationships.append({
            "segment_1": SEGMENT_NAMES.get(src_id, f"segment_{src_id}"),
            "segment_2": SEGMENT_NAMES.get(dst_id, f"segment_{dst_id}"),
            "segment_1_id": src_id,
            "segment_2_id": dst_id,
            "importance_score": round(score, 4),
            "causal": explainer_success,
        })

    return sorted(relationships, key=lambda r: r["importance_score"], reverse=True)


def generate_json_report(
    classification: str,
    confidence: float,
    segment_info: list[dict],
    edge_index: np.ndarray,
    edge_mask: np.ndarray,
    segment_ids: list[int],
    explainer_success: bool,
    image_path: str = "",
    explainer_method: str = "GNNExplainer",
) -> dict:
    """Generate a structured JSON report."""
    is_fake = classification.upper() == "FAKE"
    relationships = _build_relationships(
        edge_index, edge_mask, segment_ids, explainer_success,
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": image_path,
        "classification": classification.upper(),
        "confidence": round(float(confidence), 4),
        "is_fake": is_fake,
        "explainer_method": explainer_method,
        "explainer_success": explainer_success,
        "detected_segments": [
            {"name": s["name"], "percentage": s["percentage"]}
            for s in segment_info
            if s["segment_id"] != 0
        ],
        "key_relationships": relationships,
        "summary": _generate_summary(classification, confidence, relationships, explainer_success),
    }


def generate_text_report(json_report: dict) -> str:
    """Generate a human-readable text report from the JSON report."""
    lines = [
        "=" * 60,
        "FORENSIC ANALYSIS REPORT",
        "=" * 60,
        f"Timestamp:      {json_report['timestamp']}",
        f"Image:          {json_report.get('image_path', 'N/A')}",
        f"Classification: {json_report['classification']}",
        f"Confidence:     {json_report['confidence']:.1%}",
        f"Explainer:      {json_report['explainer_method']}",
        f"Explainer OK:   {'Yes' if json_report['explainer_success'] else 'NO — results may not be causal'}",
        "",
        "-" * 60,
        "DETECTED FACIAL SEGMENTS",
        "-" * 60,
    ]

    for seg in json_report["detected_segments"][:10]:
        lines.append(f"  • {seg['name']:<20s} {seg['percentage']:>6.2f}%")

    lines.extend([
        "",
        "-" * 60,
        "KEY RELATIONSHIPS (by importance)",
        "-" * 60,
    ])

    for rel in json_report["key_relationships"][:10]:
        score = rel["importance_score"]
        bar = "█" * int(score * 20)
        causal_tag = "" if rel["causal"] else " [attention-based]"
        lines.append(
            f"  {rel['segment_1']:<15s} ↔ {rel['segment_2']:<15s}  "
            f"{score:.4f}  {bar}{causal_tag}"
        )

    lines.extend([
        "",
        "-" * 60,
        "SUMMARY",
        "-" * 60,
        json_report["summary"],
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def _generate_summary(
    classification: str,
    confidence: float,
    relationships: list[dict],
    explainer_success: bool,
) -> str:
    """Generate a one-line summary."""
    if not explainer_success:
        return (
            f"Image classified as {classification} ({confidence:.1%} confidence). "
            f"GNNExplainer failed — edge importance scores are NOT causally grounded."
        )

    if classification.upper() == "REAL":
        return (
            f"Image classified as REAL ({confidence:.1%} confidence). "
            f"No significant anomalies detected across facial region boundaries."
        )

    # FAKE — highlight top relationships
    top = relationships[:3]
    if not top:
        return (
            f"Image classified as FAKE ({confidence:.1%} confidence). "
            f"No edge relationships available for explanation."
        )

    pairs = ", ".join(
        f"{r['segment_1']}↔{r['segment_2']} ({r['importance_score']:.3f})"
        for r in top
    )
    return (
        f"Image classified as FAKE ({confidence:.1%} confidence). "
        f"Highest edge importance: {pairs}."
    )


def save_report(
    json_report: dict,
    output_dir: str,
    image_name: str,
) -> tuple[str, str]:
    """Save both JSON and text reports to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = str(out / f"{image_name}_result.json")
    txt_path  = str(out / f"{image_name}_explanation.txt")

    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    with open(txt_path, "w") as f:
        f.write(generate_text_report(json_report))

    return json_path, txt_path
