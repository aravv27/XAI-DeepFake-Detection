"""
utils/explanation.py — v2
GNNExplainer-based causal edge attribution + report generation.

Contract (from v2_implementation_plan.md §4.8):
  - Method: GNNExplainer (PyG built-in) — REPLACES template engine entirely
  - Edge masks reflect actual decision contribution, not raw attention weights
  - Output formats: .txt forensic report, .json structured data
  - Template fallback: NONE. If GNNExplainer fails → report flagged incomplete
  - NO threshold-to-template logic anywhere
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from models.face_parser import SEGMENT_NAMES


# ---------------------------------------------------------------------------
# GNNExplainer wrapper
# ---------------------------------------------------------------------------

def explain_graph(
    model: torch.nn.Module,
    data: Data,
    target_class: int = 1,
    epochs: int = 200,
    lr: float = 0.01,
) -> dict:
    """
    Run GNNExplainer on a single graph to get causal edge masks.

    Args:
        model:        GATExplainer model (eval mode).
        data:         PyG Data with x, edge_index, y.
        target_class: Class to explain (1 = fake).
        epochs:       GNNExplainer optimisation epochs.
        lr:           GNNExplainer learning rate.

    Returns:
        Dict with 'edge_mask' (numpy), 'node_mask' (numpy), 'success' (bool).
    """
    try:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs, lr=lr),
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
            target=torch.tensor([target_class]),
            batch=torch.zeros(data.x.size(0), dtype=torch.long),
        )

        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        node_mask = explanation.node_mask.detach().cpu().numpy()

        return {
            "edge_mask": edge_mask,
            "node_mask": node_mask,
            "success": True,
        }

    except Exception as e:
        # v2 contract: never fabricate — flag as incomplete
        return {
            "edge_mask": np.zeros(data.edge_index.shape[1], dtype=np.float32),
            "node_mask": np.zeros((data.x.shape[0], data.x.shape[1]), dtype=np.float32),
            "success": False,
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
            "causal": explainer_success,  # only causal if GNNExplainer succeeded
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
) -> dict:
    """
    Generate a structured JSON report.

    Returns:
        Dict matching the schema from v2 plan §4.8.
    """
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
        "explainer_method": "GNNExplainer",
        "explainer_success": explainer_success,
        "detected_segments": [
            {"name": s["name"], "percentage": s["percentage"]}
            for s in segment_info
            if s["segment_id"] != 0  # skip background
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
        "KEY RELATIONSHIPS (by causal importance)",
        "-" * 60,
    ])

    for rel in json_report["key_relationships"][:10]:
        score = rel["importance_score"]
        bar = "█" * int(score * 20)
        causal_tag = "" if rel["causal"] else " [non-causal]"
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
        f"Highest causal edge importance: {pairs}."
    )


def save_report(
    json_report: dict,
    output_dir: str,
    image_name: str,
) -> tuple[str, str]:
    """
    Save both JSON and text reports to disk.

    Returns:
        (json_path, txt_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = str(out / f"{image_name}_result.json")
    txt_path  = str(out / f"{image_name}_explanation.txt")

    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    with open(txt_path, "w") as f:
        f.write(generate_text_report(json_report))

    return json_path, txt_path
