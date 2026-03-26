"""
utils/visualization.py — v2
Heatmap, segmentation, and graph rendering for the forensic dashboard.

Outputs (from v2_implementation_plan.md §4.8):
  - 4-panel dashboard .png: original | heatmap overlay | segment map | text summary
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional

from models.face_parser import SEGMENT_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay a heatmap on an image.

    Args:
        image:    (H, W, 3) uint8 RGB.
        heatmap:  (H, W) float32 in [0, 1].
        alpha:    Blending factor.
        colormap: Matplotlib colormap name.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    cmap = plt.get_cmap(colormap)
    coloured = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * coloured + (1 - alpha) * image).astype(np.uint8)
    return blended


def colourize_segments(
    segment_map: np.ndarray,
    image: Optional[np.ndarray] = None,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Create a colour-coded segment map, optionally blended with original image.

    Args:
        segment_map: (H, W) uint8, values 0-18.
        image:       (H, W, 3) uint8 RGB. If provided, blend with it.
        alpha:       Blending factor for segment colours.

    Returns:
        (H, W, 3) uint8 coloured image.
    """
    # Deterministic palette
    np.random.seed(42)
    palette = np.random.randint(60, 240, size=(19, 3), dtype=np.uint8)
    palette[0] = [30, 30, 30]  # background → dark grey

    h, w = segment_map.shape
    coloured = palette[segment_map.flatten()].reshape(h, w, 3)

    if image is not None:
        coloured = (alpha * coloured.astype(np.float32) +
                    (1 - alpha) * image.astype(np.float32)).astype(np.uint8)
    return coloured


# ---------------------------------------------------------------------------
# 4-panel forensic dashboard
# ---------------------------------------------------------------------------

def create_dashboard(
    image: np.ndarray,
    heatmap: np.ndarray,
    segment_map: np.ndarray,
    json_report: dict,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Create the 4-panel forensic dashboard (v2 plan §4.8).

    Panels:
      [0,0] Original Image
      [0,1] LayerCAM Heatmap Overlay
      [1,0] Face Segmentation Map
      [1,1] Analysis Summary Text

    Args:
        image:       (H, W, 3) uint8 RGB.
        heatmap:     (H, W) float32 in [0, 1].
        segment_map: (H, W) uint8.
        json_report: Report dict from generate_json_report().
        save_path:   If given, save the figure here.
        show:        If True, call plt.show().

    Returns:
        save_path if saved, else None.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Forensic Analysis: {json_report['classification']} "
        f"({json_report['confidence']:.1%})",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # --- Panel 1: Original ---
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=12)
    axes[0, 0].axis("off")

    # --- Panel 2: Heatmap overlay ---
    heatmap_overlay = overlay_heatmap(image, heatmap, alpha=0.5)
    axes[0, 1].imshow(heatmap_overlay)
    axes[0, 1].set_title("LayerCAM Attention", fontsize=12)
    axes[0, 1].axis("off")

    # --- Panel 3: Segmentation ---
    seg_coloured = colourize_segments(segment_map, image, alpha=0.6)
    axes[1, 0].imshow(seg_coloured)
    axes[1, 0].set_title("Face Segmentation", fontsize=12)
    axes[1, 0].axis("off")

    # Add segment legend (only present segments)
    present_ids = np.unique(segment_map)
    np.random.seed(42)
    palette = np.random.randint(60, 240, size=(19, 3), dtype=np.uint8)
    palette[0] = [30, 30, 30]
    legend_patches = []
    for sid in sorted(present_ids):
        if sid == 0:
            continue
        colour = palette[sid] / 255.0
        name = SEGMENT_NAMES.get(int(sid), f"seg_{sid}")
        legend_patches.append(mpatches.Patch(color=colour, label=name))
    if legend_patches:
        axes[1, 0].legend(
            handles=legend_patches, loc="lower left",
            fontsize=7, ncol=2, framealpha=0.7,
        )

    # --- Panel 4: Text summary ---
    axes[1, 1].axis("off")
    summary_lines = [
        f"Classification: {json_report['classification']}",
        f"Confidence: {json_report['confidence']:.1%}",
        f"Explainer: {json_report['explainer_method']}",
        f"Explainer OK: {'Yes' if json_report['explainer_success'] else 'NO'}",
        "",
        "Top Relationships:",
    ]
    for rel in json_report.get("key_relationships", [])[:5]:
        tag = "causal" if rel.get("causal") else "non-causal"
        summary_lines.append(
            f"  {rel['segment_1']} ↔ {rel['segment_2']}: "
            f"{rel['importance_score']:.4f} [{tag}]"
        )
    summary_lines.extend(["", "Summary:", json_report.get("summary", "")])

    axes[1, 1].text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=axes[1, 1].transAxes,
        fontsize=10, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.85),
        color="white",
    )
    axes[1, 1].set_title("Analysis Summary", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    saved_path = None
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        saved_path = save_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path
