"""
models/gat_explainer.py — v2
Graph Attention Network explainer + NodeFeatureExtractor.

Contract (from v2_implementation_plan.md §4.5, §4.7):
  NodeFeatureExtractor:
    - ALL 10 dims populated for every node. Zero-padding is FORBIDDEN.
    - Dims: mean_attn, std_attn, max_attn, area_ratio,
            laplacian_mean, laplacian_std, mean_R, mean_G,
            centroid_x, centroid_y
    - If a feature cannot be computed for a segment, that segment is EXCLUDED.

  GATExplainer:
    - Backend: PyTorch Geometric GATConv — NO manual fallback
    - Input dim: 10
    - Layer 1: GATConv(10 → 64, heads=4, concat=True) → 256
    - Layer 2: GATConv(256 → 64, heads=1, concat=False) → 64
    - Pooling: global_mean_pool → (64,)
    - Classifier: Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→2)
    - BatchNorm after each GAT layer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import laplace

# Hard requirement — no manual GATLayer fallback (v2 §10)
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch


# ---------------------------------------------------------------------------
# Node Feature Extractor — 10 fully-populated dimensions
# ---------------------------------------------------------------------------

class NodeFeatureExtractor:
    """
    Extracts a 10-dim feature vector for each facial segment.

    Every dimension is fully computed — no zero-padding.
    If a segment has zero pixels, it is excluded from the graph entirely.

    Feature vector layout (v2 plan §4.5):
      [0] mean LayerCAM attention in segment
      [1] std  LayerCAM attention in segment
      [2] max  LayerCAM attention in segment
      [3] area ratio  (segment pixels / total pixels)
      [4] Laplacian noise mean  (texture frequency)
      [5] Laplacian noise std
      [6] mean R channel in segment
      [7] mean G channel in segment
      [8] centroid X  (normalised 0-1)
      [9] centroid Y  (normalised 0-1)
    """

    FEATURE_DIM = 10

    def extract(
        self,
        image: np.ndarray,
        segment_map: np.ndarray,
        attention_map: np.ndarray,
        segment_ids: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        """
        Extract features for the given segment IDs.

        Args:
            image:         (H, W, 3) uint8 RGB image.
            segment_map:   (H, W) uint8 segment map (values 0-18).
            attention_map: (H, W) float32 LayerCAM heatmap in [0, 1].
            segment_ids:   List of segment IDs to extract features for
                           (background=0 should NOT be included).

        Returns:
            features:      (N, 10) float32 array where N ≤ len(segment_ids).
            valid_ids:     List of segment IDs that were successfully extracted
                           (segments with 0 pixels are excluded).
        """
        h, w = segment_map.shape
        total_pixels = h * w

        # Pre-compute greyscale Laplacian (texture frequency map)
        grey = np.mean(image.astype(np.float32), axis=2)  # (H, W)
        lap = np.abs(laplace(grey))  # (H, W) absolute Laplacian

        features = []
        valid_ids = []

        for sid in segment_ids:
            mask = segment_map == sid
            pixel_count = mask.sum()
            if pixel_count == 0:
                # v2 contract: exclude segments with 0 pixels
                continue

            # [0-2] Attention statistics
            attn_vals = attention_map[mask]
            mean_attn = float(attn_vals.mean())
            std_attn  = float(attn_vals.std())
            max_attn  = float(attn_vals.max())

            # [3] Area ratio
            area_ratio = float(pixel_count) / total_pixels

            # [4-5] Laplacian noise (texture frequency)
            lap_vals = lap[mask]
            lap_mean = float(lap_vals.mean())
            lap_std  = float(lap_vals.std())

            # [6-7] Mean colour channels
            pixels = image[mask]  # (K, 3)
            mean_r = float(pixels[:, 0].mean()) / 255.0
            mean_g = float(pixels[:, 1].mean()) / 255.0

            # [8-9] Centroid (normalised)
            ys, xs = np.where(mask)
            centroid_x = float(xs.mean()) / w
            centroid_y = float(ys.mean()) / h

            feat = np.array([
                mean_attn, std_attn, max_attn,
                area_ratio,
                lap_mean, lap_std,
                mean_r, mean_g,
                centroid_x, centroid_y,
            ], dtype=np.float32)

            # Final guard: no NaN allowed (v2 integration contract §5.1)
            if np.any(np.isnan(feat)):
                continue

            features.append(feat)
            valid_ids.append(sid)

        if features:
            return np.stack(features, axis=0), valid_ids
        else:
            return np.zeros((0, self.FEATURE_DIM), dtype=np.float32), []


# ---------------------------------------------------------------------------
# GATExplainer — PyTorch Geometric only
# ---------------------------------------------------------------------------

class GATExplainer(nn.Module):
    """
    Graph Attention Network for deepfake edge-level explanation.

    Architecture (v2 plan §4.7):
      GATConv(10→64, heads=4) → BN → ELU
      GATConv(256→64, heads=1) → BN → ELU
      global_mean_pool → (64,)
      Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→2)
    """

    def __init__(self, input_dim: int = 10, num_classes: int = 2):
        super().__init__()
        if input_dim != NodeFeatureExtractor.FEATURE_DIM:
            raise ValueError(
                f"GATExplainer input_dim must be {NodeFeatureExtractor.FEATURE_DIM}, "
                f"got {input_dim}"
            )

        # GAT layers
        self.conv1 = GATConv(input_dim, 64, heads=4, concat=True, dropout=0.3)
        self.bn1   = nn.BatchNorm1d(256)  # 64 * 4 heads
        self.conv2 = GATConv(256, 64, heads=1, concat=False, dropout=0.3)
        self.bn2   = nn.BatchNorm1d(64)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple]:
        """
        Forward pass.

        Args:
            x:           (N_total, 10) node features.
            edge_index:  (2, E_total) edge indices.
            batch:       (N_total,) batch assignment vector.
            return_attention: if True, return GAT attention weights.

        Returns:
            logits (B, 2),  or  (logits, (attn1, attn2)).
        """
        # Layer 1
        if return_attention:
            x1, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        else:
            x1 = self.conv1(x, edge_index)
            attn1 = None
        x1 = self.bn1(x1)
        x1 = F.elu(x1)

        # Layer 2
        if return_attention:
            x2, attn2 = self.conv2(x1, edge_index, return_attention_weights=True)
        else:
            x2 = self.conv2(x1, edge_index)
            attn2 = None
        x2 = self.bn2(x2)
        x2 = F.elu(x2)

        # Pooling → graph-level vector
        graph_vec = global_mean_pool(x2, batch)  # (B, 64)

        # Classify
        logits = self.classifier(graph_vec)  # (B, 2)

        if return_attention:
            return logits, (attn1, attn2)
        return logits


# ---------------------------------------------------------------------------
# Graph batching helper
# ---------------------------------------------------------------------------

def build_pyg_data(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    label: int,
) -> Data:
    """
    Create a PyG Data object for one image's face graph.

    Args:
        node_features: (N, 10) float32 array.
        edge_index:    (2, E) int64 array.
        label:         0 (real) or 1 (fake).

    Returns:
        PyG Data object.
    """
    return Data(
        x=torch.from_numpy(node_features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        y=torch.tensor([label], dtype=torch.long),
    )


def create_gat_batch(data_list: list[Data]) -> Batch:
    """
    Batch multiple PyG Data objects into a single Batch.

    Uses PyG's Batch.from_data_list — no manual offset logic.
    """
    return Batch.from_data_list(data_list)
