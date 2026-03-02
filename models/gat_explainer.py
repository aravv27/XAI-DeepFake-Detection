"""
Graph Attention Network (GAT) Explainer for Deepfake Detection

Uses Graph Attention Networks to model relationships between facial segments
and provide interpretable explanations for deepfake detection decisions.

The GAT learns which segment relationships (edges) are most indicative of
manipulation, providing human-readable explanations.

Reference:
- Veličković et al. (2018). Graph Attention Networks
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

# Try importing PyTorch Geometric
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. Using fallback GAT implementation.")


class GATLayer(nn.Module):
    """
    Manual GAT layer implementation (fallback if PyG not available).
    
    Implements multi-head graph attention as described in Veličković et al.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        
        # Attention parameters
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features (N, in_features)
            edge_index: Edge indices (2, E)
            return_attention: If True, return attention weights
        
        Returns:
            out: Updated node features
            attention (optional): Edge attention weights (E, num_heads)
        """
        N = x.size(0)
        
        # Apply linear transformation: (N, in_features) -> (num_heads, N, out_features)
        x = x.unsqueeze(0).expand(self.num_heads, -1, -1)
        Wh = torch.bmm(x, self.W)  # (num_heads, N, out_features)
        
        # Compute attention scores
        src, dst = edge_index[0], edge_index[1]
        
        # Source and destination attention
        e_src = torch.bmm(Wh, self.a_src).squeeze(-1)  # (num_heads, N)
        e_dst = torch.bmm(Wh, self.a_dst).squeeze(-1)  # (num_heads, N)
        
        # Attention for each edge
        e = e_src[:, src] + e_dst[:, dst]  # (num_heads, E)
        e = self.leaky_relu(e)
        
        # Softmax over neighbors
        attention = self._edge_softmax(e, dst, N)  # (num_heads, E)
        attention = self.dropout(attention)
        
        # Aggregate neighbors
        out = torch.zeros(self.num_heads, N, self.out_features, device=x.device)
        for head in range(self.num_heads):
            for i, (s, d) in enumerate(edge_index.T):
                out[head, d] += attention[head, i] * Wh[head, s]
        
        if self.concat:
            out = out.permute(1, 0, 2).reshape(N, -1)  # (N, num_heads * out_features)
        else:
            out = out.mean(dim=0)  # (N, out_features)
        
        if return_attention:
            return out, attention.permute(1, 0)  # (E, num_heads)
        return out
    
    def _edge_softmax(
        self,
        e: torch.Tensor,
        dst: torch.Tensor,
        N: int
    ) -> torch.Tensor:
        """Compute softmax over edges for each destination node."""
        # Numerical stability
        e_max = torch.zeros(self.num_heads, N, device=e.device)
        for head in range(self.num_heads):
            for i, d in enumerate(dst):
                e_max[head, d] = max(e_max[head, d], e[head, i])
        
        e = e - e_max[:, dst]
        exp_e = torch.exp(e)
        
        # Sum for normalization
        exp_sum = torch.zeros(self.num_heads, N, device=e.device)
        for head in range(self.num_heads):
            for i, d in enumerate(dst):
                exp_sum[head, d] += exp_e[head, i]
        
        return exp_e / (exp_sum[:, dst] + 1e-8)


class GATExplainer(nn.Module):
    """
    Graph Attention Network for deepfake segment relationship modeling.
    
    Takes node features (per facial segment) and learns which segment
    relationships (edges) are most suspicious for detecting manipulations.
    
    Node Features (Hybrid - 262 dimensions):
    - mean_attention (1): Average attention value in segment
    - std_attention (1): Attention variance
    - max_attention (1): Maximum attention value
    - area_ratio (1): Segment size relative to face
    - cnn_features (256): Pooled CNN features from segment
    - texture_features (2): Noise statistics (mean, std of high-freq)
    
    Args:
        in_features: Node feature dimension (default: 262)
        hidden: Hidden layer dimension
        heads: Number of attention heads
        num_classes: Output classes (2 for real/fake)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int = 262,
        hidden: int = 128,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.heads = heads
        
        if PYG_AVAILABLE:
            self.gat1 = GATConv(
                in_features, hidden,
                heads=heads,
                dropout=dropout,
                concat=True
            )
            self.gat2 = GATConv(
                hidden * heads, hidden,
                heads=1,
                dropout=dropout,
                concat=False
            )
        else:
            self.gat1 = GATLayer(in_features, hidden, heads, dropout, concat=True)
            self.gat2 = GATLayer(hidden * heads, hidden, 1, dropout, concat=False)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )
        
        # Edge importance predictor (for explainability)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        
        print(f"[GATExplainer] in_features={in_features}, hidden={hidden}, heads={heads}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GAT.
        
        Args:
            x: Node features (N, in_features)
            edge_index: Edge indices (2, E)
            batch: Batch assignment for nodes (N,) - for batched graphs
            return_attention: If True, compute edge importance scores
        
        Returns:
            Dict with:
            - logits: Classification logits (B, num_classes)
            - node_features: Final node representations (N, hidden)
            - edge_importance: Edge importance scores (E,) if return_attention=True
        """
        # First GAT layer
        if PYG_AVAILABLE:
            h1, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            h1, attn1 = self.gat1(x, edge_index, return_attention=True)
        
        h1 = F.elu(h1)
        h1 = self.bn1(h1)
        h1 = self.dropout(h1)
        
        # Second GAT layer
        if PYG_AVAILABLE:
            h2, attn2 = self.gat2(h1, edge_index, return_attention_weights=True)
        else:
            h2, attn2 = self.gat2(h1, edge_index, return_attention=True)
        
        h2 = F.elu(h2)
        h2 = self.bn2(h2)
        
        # Global pooling for graph-level representation
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if PYG_AVAILABLE:
            graph_repr = global_mean_pool(h2, batch)
        else:
            # Manual global mean pooling
            num_graphs = batch.max().item() + 1
            graph_repr = torch.zeros(num_graphs, h2.size(1), device=h2.device)
            count = torch.zeros(num_graphs, device=h2.device)
            for i, b in enumerate(batch):
                graph_repr[b] += h2[i]
                count[b] += 1
            graph_repr = graph_repr / count.unsqueeze(1).clamp(min=1)
        
        # Classification
        logits = self.classifier(graph_repr)
        
        result = {
            'logits': logits,
            'node_features': h2
        }
        
        # Compute edge importance for explainability
        if return_attention:
            edge_importance = self._compute_edge_importance(h2, edge_index)
            result['edge_importance'] = edge_importance
            result['gat_attention'] = attn2 if isinstance(attn2, torch.Tensor) else attn2[1]
        
        return result
    
    def _compute_edge_importance(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance score for each edge.
        
        Higher scores indicate more suspicious segment relationships.
        """
        src, dst = edge_index
        
        # Concatenate source and destination features
        edge_features = torch.cat([
            node_features[src],
            node_features[dst]
        ], dim=1)
        
        # Predict edge importance
        importance = self.edge_predictor(edge_features).squeeze(-1)
        
        return importance
    
    def get_explanation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        segment_names: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Generate human-readable explanation of suspicious relationships.
        
        Args:
            x: Node features
            edge_index: Edge indices
            segment_names: Names of segments corresponding to nodes
            top_k: Number of top suspicious relationships to return
        
        Returns:
            List of dicts with segment pairs and importance scores
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(x, edge_index, return_attention=True)
        
        edge_importance = result['edge_importance'].cpu().numpy()
        
        # Get top-k suspicious edges
        top_indices = np.argsort(edge_importance)[::-1][:top_k]
        
        explanations = []
        for idx in top_indices:
            src, dst = edge_index[:, idx].cpu().numpy()
            explanations.append({
                'segment_1': segment_names[src],
                'segment_2': segment_names[dst],
                'importance': float(edge_importance[idx]),
                'suspicious': edge_importance[idx] > 0.5
            })
        
        return explanations


class NodeFeatureExtractor:
    """
    Extract node features for GAT from attention maps and CNN features.
    
    For each facial segment, extracts:
    - Attention statistics (mean, std, max)
    - Area ratio
    - Pooled CNN features
    - Texture features (high-frequency noise analysis)
    """
    
    def __init__(
        self,
        cnn_feature_dim: int = 256,
        texture_feature_dim: int = 2
    ):
        self.cnn_feature_dim = cnn_feature_dim
        self.texture_feature_dim = texture_feature_dim
        
        # Feature dimension: 4 (attention stats) + cnn + texture
        self.feature_dim = 4 + cnn_feature_dim + texture_feature_dim
        print(f"[NodeFeatureExtractor] Feature dim: {self.feature_dim}")
    
    def extract(
        self,
        attention_map: np.ndarray,
        segment_map: np.ndarray,
        cnn_features: torch.Tensor,
        image: np.ndarray,
        segment_ids: List[int]
    ) -> torch.Tensor:
        """
        Extract features for each segment.
        
        Args:
            attention_map: (H, W) attention heatmap
            segment_map: (H, W) segment IDs
            cnn_features: (C, H', W') CNN feature map
            image: (H, W, 3) original image
            segment_ids: List of segment IDs to extract features for
        
        Returns:
            node_features: (num_segments, feature_dim) tensor
        """
        features = []
        
        for seg_id in segment_ids:
            mask = (segment_map == seg_id).astype(np.float32)
            
            if mask.sum() == 0:
                # Empty segment - use zeros
                feat = np.zeros(self.feature_dim)
            else:
                feat = self._extract_segment_features(
                    attention_map, mask, cnn_features, image
                )
            
            features.append(feat)
        
        return torch.tensor(np.array(features), dtype=torch.float32)
    
    def _extract_segment_features(
        self,
        attention_map: np.ndarray,
        mask: np.ndarray,
        cnn_features: torch.Tensor,
        image: np.ndarray
    ) -> np.ndarray:
        """Extract features for a single segment."""
        features = []
        
        # 1. Attention statistics
        masked_attention = attention_map * mask
        nonzero = mask > 0
        if nonzero.sum() > 0:
            attn_values = attention_map[nonzero]
            features.extend([
                attn_values.mean(),  # Mean attention
                attn_values.std(),   # Std attention
                attn_values.max(),   # Max attention
            ])
        else:
            features.extend([0, 0, 0])
        
        # 2. Area ratio
        area_ratio = mask.sum() / mask.size
        features.append(area_ratio)
        
        # 3. Pooled CNN features
        cnn_feat = self._pool_cnn_features(cnn_features, mask)
        features.extend(cnn_feat.tolist())
        
        # 4. Texture features (high-frequency analysis)
        texture_feat = self._extract_texture_features(image, mask)
        features.extend(texture_feat)
        
        return np.array(features)
    
    def _pool_cnn_features(
        self,
        cnn_features: torch.Tensor,
        mask: np.ndarray
    ) -> np.ndarray:
        """Pool CNN features within segment mask."""
        # Resize mask to feature map size
        _, h, w = cnn_features.shape
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
        ).astype(np.float32)
        
        mask_tensor = torch.from_numpy(mask_resized).to(cnn_features.device)
        
        # Masked average pooling
        if mask_tensor.sum() > 0:
            masked = cnn_features * mask_tensor.unsqueeze(0)
            pooled = masked.sum(dim=(1, 2)) / mask_tensor.sum()
        else:
            pooled = torch.zeros(cnn_features.size(0))
        
        # Reduce to target dimension if needed
        pooled = pooled.cpu().numpy()
        if len(pooled) > self.cnn_feature_dim:
            # Simple average reduction
            step = len(pooled) // self.cnn_feature_dim
            pooled = pooled[::step][:self.cnn_feature_dim]
        elif len(pooled) < self.cnn_feature_dim:
            # Pad with zeros
            pooled = np.pad(pooled, (0, self.cnn_feature_dim - len(pooled)))
        
        return pooled
    
    def _extract_texture_features(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> List[float]:
        """Extract texture/noise features from segment."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image
        
        # High-pass filter (Laplacian) for noise detection
        from scipy import ndimage
        laplacian = ndimage.laplace(gray.astype(np.float32))
        
        # Masked statistics
        masked_lap = laplacian * mask
        nonzero = mask > 0
        
        if nonzero.sum() > 0:
            noise_mean = np.abs(masked_lap[nonzero]).mean()
            noise_std = np.abs(masked_lap[nonzero]).std()
        else:
            noise_mean, noise_std = 0, 0
        
        return [noise_mean, noise_std]


# Import PIL for resizing
from PIL import Image


def create_gat_batch(
    node_features_list: List[torch.Tensor],
    edge_index_list: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch multiple graphs for GAT forward pass.
    
    Args:
        node_features_list: List of (N_i, F) node feature tensors
        edge_index_list: List of (2, E_i) edge index tensors
    
    Returns:
        batched_features: (sum(N_i), F)
        batched_edges: (2, sum(E_i))
        batch: (sum(N_i),) batch assignment
    """
    if PYG_AVAILABLE:
        data_list = [
            Data(x=nf, edge_index=ei) 
            for nf, ei in zip(node_features_list, edge_index_list)
        ]
        batch = Batch.from_data_list(data_list)
        return batch.x, batch.edge_index, batch.batch
    
    # Manual batching
    all_features = []
    all_edges = []
    batch_indices = []
    
    node_offset = 0
    for i, (nf, ei) in enumerate(zip(node_features_list, edge_index_list)):
        all_features.append(nf)
        all_edges.append(ei + node_offset)
        batch_indices.extend([i] * nf.size(0))
        node_offset += nf.size(0)
    
    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_edges, dim=1),
        torch.tensor(batch_indices, dtype=torch.long)
    )
