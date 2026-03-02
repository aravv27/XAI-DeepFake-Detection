"""
Face Parser using BiSeNet

BiSeNet-based face parsing model trained on CelebAMask-HQ dataset.
Provides 19 facial segment classes including skin, eyes, nose, mouth, hair, etc.

Reference:
- Yu et al. (2018). BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation
- Lee et al. (2020). MaskGAN: CelebAMask-HQ Dataset

Pretrained weights from: https://github.com/zllrunning/face-parsing.PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as T


# Facial segment classes from CelebAMask-HQ
FACE_SEGMENTS = [
    'background',   # 0
    'skin',         # 1
    'left_brow',    # 2
    'right_brow',   # 3
    'left_eye',     # 4
    'right_eye',    # 5
    'eye_glasses',  # 6
    'left_ear',     # 7
    'right_ear',    # 8
    'earring',      # 9
    'nose',         # 10
    'mouth',        # 11
    'upper_lip',    # 12
    'lower_lip',    # 13
    'neck',         # 14
    'necklace',     # 15
    'cloth',        # 16
    'hair',         # 17
    'hat'           # 18
]

# Grouped segments for analysis
SEGMENT_GROUPS = {
    'eyes': [4, 5, 6],  # left_eye, right_eye, glasses
    'brows': [2, 3],    # left_brow, right_brow  
    'nose': [10],
    'mouth': [11, 12, 13],  # mouth, upper_lip, lower_lip
    'ears': [7, 8, 9],  # left_ear, right_ear, earring
    'skin': [1],
    'hair': [17, 18],   # hair, hat
    'neck': [14, 15, 16],  # neck, necklace, cloth
}

# Color palette for visualization (RGB)
SEGMENT_COLORS = [
    (0, 0, 0),       # background - black
    (255, 220, 185), # skin - peach
    (139, 69, 19),   # left_brow - brown
    (139, 69, 19),   # right_brow - brown
    (0, 191, 255),   # left_eye - deep sky blue
    (0, 191, 255),   # right_eye - deep sky blue
    (70, 130, 180),  # eye_glasses - steel blue
    (255, 182, 193), # left_ear - light pink
    (255, 182, 193), # right_ear - light pink
    (255, 215, 0),   # earring - gold
    (255, 99, 71),   # nose - tomato
    (220, 20, 60),   # mouth - crimson
    (255, 105, 180), # upper_lip - hot pink
    (255, 20, 147),  # lower_lip - deep pink
    (210, 180, 140), # neck - tan
    (192, 192, 192), # necklace - silver
    (100, 149, 237), # cloth - cornflower blue
    (139, 90, 43),   # hair - sienna
    (128, 0, 128),   # hat - purple
]


# BiSeNet Components
class ConvBNReLU(nn.Module):
    """Conv-BatchNorm-ReLU block."""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialPath(nn.Module):
    """Spatial path for preserving spatial information."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, stride=2)
        self.conv2 = ConvBNReLU(64, 128, stride=2)
        self.conv3 = ConvBNReLU(128, 256, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class AttentionRefinementModule(nn.Module):
    """ARM for refining context features."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        self.conv_atten = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_ch)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid(atten)
        return feat * atten


class FeatureFusionModule(nn.Module):
    """FFM for fusing spatial and context paths."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.convblk = ConvBNReLU(in_ch, out_ch, 1, 1, 0)
        self.conv1 = nn.Conv2d(out_ch, out_ch // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch // 4, out_ch, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        return feat + feat * atten


class ContextPath(nn.Module):
    """Context path with ResNet18 backbone."""
    
    def __init__(self):
        super().__init__()
        # Simplified ResNet-like backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        
        self.conv_head32 = ConvBNReLU(128, 128, 3, 1, 1)
        self.conv_head16 = ConvBNReLU(128, 128, 3, 1, 1)
        
        self.conv_avg = ConvBNReLU(512, 128, 1, 1, 0)
    
    def _make_layer(
        self, 
        in_ch: int, 
        out_ch: int, 
        num_blocks: int, 
        stride: int = 1
    ):
        layers = []
        layers.append(self._basic_block(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(self._basic_block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def _basic_block(self, in_ch: int, out_ch: int, stride: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    
    def forward(self, x):
        feat8 = self.conv1(x)
        feat8 = self.maxpool(feat8)
        feat8 = self.layer1(feat8)
        feat8 = self.layer2(feat8)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        
        avg = F.adaptive_avg_pool2d(feat32, 1)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, feat32.shape[2:], mode='nearest')
        
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, feat16.shape[2:], mode='nearest')
        feat32_up = self.conv_head32(feat32_up)
        
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, feat8.shape[2:], mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        
        return feat16_up, feat32_up


class BiSeNet(nn.Module):
    """BiSeNet for face parsing."""
    
    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256 + 128, 256)
        self.conv_out = nn.Conv2d(256, num_classes, 1, bias=False)
    
    def forward(self, x):
        feat_sp = self.sp(x)
        feat_cp, _ = self.cp(x)
        
        feat_cp = F.interpolate(
            feat_cp, feat_sp.shape[2:], mode='bilinear', align_corners=True
        )
        feat_fuse = self.ffm(feat_sp, feat_cp)
        
        out = self.conv_out(feat_fuse)
        out = F.interpolate(
            out, x.shape[2:], mode='bilinear', align_corners=True
        )
        return out


class FaceParser:
    """
    High-level face parsing interface.
    
    Parses face images into 19 semantic segments.
    Uses BiSeNet trained on CelebAMask-HQ.
    
    Args:
        device: Device to run inference on
        checkpoint_path: Path to pretrained weights (optional)
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        self.model = BiSeNet(num_classes=19).to(self.device)
        
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Transform for input images
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.segment_names = FACE_SEGMENTS
        self.segment_colors = SEGMENT_COLORS
        
        print(f"[FaceParser] Loaded on {self.device}")
    
    def _load_checkpoint(self, path: str):
        """Load pretrained weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"[FaceParser] Loaded checkpoint from {path}")
    
    @torch.no_grad()
    def parse(
        self, 
        image: Image.Image | np.ndarray,
        return_probs: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Parse face image into segments.
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
            return_probs: If True, also return class probabilities
        
        Returns:
            segment_map: (H, W) array of segment IDs (0-18)
            probs (optional): (19, H, W) array of class probabilities
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_size = image.size  # (W, H)
        
        # Transform and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)  # (1, 19, H, W)
        
        # Get probabilities and predictions
        probs = F.softmax(output, dim=1)[0]  # (19, H, W)
        segment_map = output.argmax(1)[0]  # (H, W)
        
        # Resize to original size
        segment_map = F.interpolate(
            segment_map.unsqueeze(0).unsqueeze(0).float(),
            size=(original_size[1], original_size[0]),
            mode='nearest'
        )[0, 0].long()
        
        segment_map = segment_map.cpu().numpy().astype(np.uint8)
        
        if return_probs:
            probs = F.interpolate(
                probs.unsqueeze(0),
                size=(original_size[1], original_size[0]),
                mode='bilinear',
                align_corners=True
            )[0].cpu().numpy()
            return segment_map, probs
        
        return segment_map
    
    def get_segment_masks(
        self, 
        segment_map: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Get individual binary masks for each segment.
        
        Args:
            segment_map: (H, W) array of segment IDs
        
        Returns:
            Dict mapping segment_id -> binary mask (H, W)
        """
        masks = {}
        for seg_id in range(len(FACE_SEGMENTS)):
            mask = (segment_map == seg_id).astype(np.float32)
            if mask.sum() > 0:
                masks[seg_id] = mask
        return masks
    
    def get_segment_info(
        self, 
        segment_map: np.ndarray
    ) -> List[Dict]:
        """
        Get information about detected segments.
        
        Returns list of dicts with:
        - segment_id, name, pixel_count, percentage, centroid
        """
        total_pixels = segment_map.size
        segments_info = []
        
        for seg_id in range(len(FACE_SEGMENTS)):
            mask = (segment_map == seg_id)
            pixel_count = mask.sum()
            
            if pixel_count > 0:
                percentage = (pixel_count / total_pixels) * 100
                
                # Find centroid
                y_indices, x_indices = np.where(mask)
                centroid = (int(x_indices.mean()), int(y_indices.mean()))
                
                segments_info.append({
                    'segment_id': seg_id,
                    'name': FACE_SEGMENTS[seg_id],
                    'pixel_count': int(pixel_count),
                    'percentage': float(percentage),
                    'centroid': centroid,
                    'mask': mask.astype(np.uint8)
                })
        
        # Sort by pixel count (largest first, excluding background)
        segments_info = sorted(
            [s for s in segments_info if s['segment_id'] != 0],
            key=lambda x: x['pixel_count'],
            reverse=True
        )
        
        return segments_info
    
    def visualize(
        self, 
        segment_map: np.ndarray, 
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create colored visualization of segments.
        
        Args:
            segment_map: (H, W) array of segment IDs
            alpha: Opacity for overlay
        
        Returns:
            colored: (H, W, 3) RGB visualization
        """
        h, w = segment_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for seg_id in range(len(FACE_SEGMENTS)):
            mask = segment_map == seg_id
            colored[mask] = SEGMENT_COLORS[seg_id]
        
        return colored


def get_face_segment_adjacency() -> List[Tuple[int, int]]:
    """
    Get anatomically plausible adjacency pairs for facial segments.
    Used to build graph edges for GAT.
    
    Returns:
        List of (segment_id_1, segment_id_2) tuples representing adjacent segments
    """
    # Anatomical adjacencies (approximate)
    adjacencies = [
        (1, 2),   # skin - left_brow
        (1, 3),   # skin - right_brow
        (1, 4),   # skin - left_eye
        (1, 5),   # skin - right_eye
        (1, 7),   # skin - left_ear
        (1, 8),   # skin - right_ear
        (1, 10),  # skin - nose
        (1, 11),  # skin - mouth
        (1, 14),  # skin - neck
        (1, 17),  # skin - hair
        (2, 4),   # left_brow - left_eye
        (3, 5),   # right_brow - right_eye
        (4, 10),  # left_eye - nose
        (5, 10),  # right_eye - nose
        (10, 11), # nose - mouth
        (10, 12), # nose - upper_lip
        (11, 12), # mouth - upper_lip
        (11, 13), # mouth - lower_lip
        (12, 13), # upper_lip - lower_lip
        (7, 17),  # left_ear - hair
        (8, 17),  # right_ear - hair
        (14, 16), # neck - cloth
    ]
    return adjacencies


def build_face_graph_edges(
    segment_map: np.ndarray,
    fully_connected: bool = True
) -> Tuple[torch.Tensor, List[int]]:
    """
    Build graph edges from segment map.
    
    Args:
        segment_map: (H, W) segment IDs
        fully_connected: If True, connect all present segments
    
    Returns:
        edge_index: (2, num_edges) tensor for PyG
        present_segments: List of segment IDs present in image
    """
    # Find present segments
    present_segments = list(np.unique(segment_map))
    if 0 in present_segments:  # Remove background
        present_segments.remove(0)
    
    edges = []
    
    if fully_connected:
        # Connect all segments
        for i, seg1 in enumerate(present_segments):
            for j, seg2 in enumerate(present_segments):
                if i != j:
                    edges.append([i, j])
    else:
        # Use anatomical adjacencies
        adjacencies = get_face_segment_adjacency()
        seg_to_idx = {s: i for i, s in enumerate(present_segments)}
        
        for s1, s2 in adjacencies:
            if s1 in seg_to_idx and s2 in seg_to_idx:
                i, j = seg_to_idx[s1], seg_to_idx[s2]
                edges.append([i, j])
                edges.append([j, i])  # Bidirectional
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return edge_index, present_segments
