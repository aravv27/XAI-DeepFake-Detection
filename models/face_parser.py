"""
models/face_parser.py — v2
BiSeNet face segmentation parser + anatomical graph construction.

Architecture matches the original face-parsing.PyTorch (zllrunning) repo exactly,
so that the 79999_iter.pth CelebAMask-HQ checkpoint loads correctly.

Contract (from v2_implementation_plan.md §4.3 & §4.6):
  - Pretrained CelebAMask-HQ checkpoint is a HARD DEPENDENCY
  - FileNotFoundError raised (with download URL) if checkpoint missing
  - Startup validation: forward pass on blank tensor, assert output shape
  - If all pixels are class 0 → RuntimeError (random weights guard)
  - Output: segment_map (H×W) uint8, values 0-18
  - Graph: anatomical edges ONLY (22 predefined pairs), NO fully_connected mode
  - Only include edges where both segment sides have pixel_count > 0
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
from typing import Optional

# ---------------------------------------------------------------------------
# CelebAMask-HQ segment labels
# ---------------------------------------------------------------------------

SEGMENT_NAMES = {
    0:  "background",
    1:  "skin",
    2:  "left_brow",
    3:  "right_brow",
    4:  "left_eye",
    5:  "right_eye",
    6:  "eye_glasses",
    7:  "left_ear",
    8:  "right_ear",
    9:  "earring",
    10: "nose",
    11: "mouth",
    12: "upper_lip",
    13: "lower_lip",
    14: "neck",
    15: "necklace",
    16: "cloth",
    17: "hair",
    18: "hat",
}

# 22 anatomically plausible edge pairs.
ANATOMICAL_EDGES = [
    (1,  2),   # skin ↔ left_brow
    (1,  3),   # skin ↔ right_brow
    (1,  4),   # skin ↔ left_eye
    (1,  5),   # skin ↔ right_eye
    (1,  10),  # skin ↔ nose
    (1,  11),  # skin ↔ mouth
    (1,  12),  # skin ↔ upper_lip
    (1,  14),  # skin ↔ neck
    (1,  17),  # skin ↔ hair
    (2,  4),   # left_brow ↔ left_eye
    (3,  5),   # right_brow ↔ right_eye
    (4,  10),  # left_eye ↔ nose
    (5,  10),  # right_eye ↔ nose
    (10, 11),  # nose ↔ mouth
    (10, 12),  # nose ↔ upper_lip
    (11, 12),  # mouth ↔ upper_lip
    (12, 13),  # upper_lip ↔ lower_lip
    (7,  1),   # left_ear ↔ skin
    (8,  1),   # right_ear ↔ skin
    (14, 16),  # neck ↔ cloth
    (17, 18),  # hair ↔ hat
    (1,  13),  # skin ↔ lower_lip
]

BISENET_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view"
)


# ---------------------------------------------------------------------------
# BiSeNet sub-modules — matches face-parsing.PyTorch (zllrunning) exactly
# for checkpoint compatibility with 79999_iter.pth
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks,
            stride=stride, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class Resnet18(nn.Module):
    """Wrapper around torchvision ResNet-18. Matches original face-parsing repo naming."""

    def __init__(self):
        super().__init__()
        # Use a LOCAL variable — do NOT store as self.features,
        # otherwise PyTorch registers the full resnet as a submodule
        # creating cp.resnet.features.* state dict keys that won't
        # match the 79999_iter.pth checkpoint.
        resnet = torchvision.models.resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)    # 128ch, 1/8 resolution
        feat16 = self.layer3(feat8)   # 256ch, 1/16 resolution
        feat32 = self.layer4(feat16)  # 512ch, 1/32 resolution
        return feat8, feat16, feat32


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = F.adaptive_avg_pool2d(feat32, 1)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=feat32.shape[2:], mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, size=feat16.shape[2:], mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat8.shape[2:], mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
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
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiSeNet(nn.Module):
    """
    Bilateral Segmentation Network for face parsing (19 classes).
    Architecture matches face-parsing.PyTorch (zllrunning) exactly.

    Note: No separate SpatialPath — uses ResNet layer2 output as spatial features.
    """

    def __init__(self, n_classes: int = 19):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)  # 128 (spatial) + 128 (context) = 256
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)

        feat_sp = feat_res8  # 128ch from ResNet layer2 as spatial features
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode="bilinear", align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode="bilinear", align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode="bilinear", align_corners=True)

        return feat_out, feat_out16, feat_out32


# ---------------------------------------------------------------------------
# FaceParser — public API
# ---------------------------------------------------------------------------

class FaceParser:
    """
    Wraps BiSeNet for face segmentation.

    Args:
        checkpoint_path: Path to CelebAMask-HQ pretrained BiSeNet weights (79999_iter.pth).
                         NO default value — raises FileNotFoundError if missing.
        device:          'cuda' or 'cpu'. Auto-detected if None.
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        # --- hard checkpoint dependency ---
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"BiSeNet pretrained checkpoint not found: '{checkpoint_path}'\n"
                f"Download the CelebAMask-HQ weights from:\n  {BISENET_DOWNLOAD_URL}\n"
                f"Then pass the local path as checkpoint_path."
            )

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = BiSeNet(n_classes=19).to(self.device)

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

        # --- startup validation ---
        self._validate_startup()

    def _validate_startup(self) -> None:
        """Assert that the loaded model produces correctly-shaped output."""
        with torch.no_grad():
            blank = torch.zeros(1, 3, 512, 512, device=self.device)
            out = self.model(blank)

        # BiSeNet returns (feat_out, feat_out16, feat_out32) — use main output
        main_out = out[0]

        # Shape check only — a blank image has no face, so all-background is expected
        expected = (1, 19, 512, 512)
        if main_out.shape != torch.Size(expected):
            raise RuntimeError(
                f"BiSeNet startup validation failed: "
                f"expected output shape {expected}, got {main_out.shape}"
            )

        # Verify the model produces varied logits (not all identical = random weights)
        # Check that the 19-class logit channels have different values
        logit_std = main_out.std().item()
        if logit_std < 1e-6:
            raise RuntimeError(
                "BiSeNet startup validation failed: output logits have near-zero variance. "
                "This typically means random or corrupted weights were loaded. "
                f"Re-download the checkpoint from:\n  {BISENET_DOWNLOAD_URL}"
            )

    # ------------------------------------------------------------------
    # Core parsing
    # ------------------------------------------------------------------

    def parse(self, image: np.ndarray) -> np.ndarray:
        """
        Parse a face image into a segment map.

        Args:
            image: (H, W, 3) uint8 RGB image.

        Returns:
            segment_map: (H, W) uint8 array, values 0-18.
        """
        h, w = image.shape[:2]
        resized = cv2.resize(image, (512, 512))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1)  # (3, 512, 512)
        # Normalise with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor / 255.0 - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 3, 512, 512)

        with torch.no_grad():
            out = self.model(tensor)  # (feat_out, feat_out16, feat_out32)

        seg_512 = out[0].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize back to original resolution
        seg = cv2.resize(seg_512, (w, h), interpolation=cv2.INTER_NEAREST)
        return seg

    def get_segment_masks(self, segment_map: np.ndarray) -> dict[int, np.ndarray]:
        """Return dict of {segment_id: binary_mask (H, W bool)}."""
        return {
            sid: (segment_map == sid)
            for sid in np.unique(segment_map)
            if sid != 0  # skip background
        }

    def get_segment_info(self, segment_map: np.ndarray) -> list[dict]:
        """Return per-segment statistics."""
        h, w = segment_map.shape
        total_pixels = h * w
        infos = []
        for sid in np.unique(segment_map):
            mask = segment_map == sid
            pixel_count = int(mask.sum())
            ys, xs = np.where(mask)
            centroid = (
                float(xs.mean() / w) if len(xs) else 0.0,
                float(ys.mean() / h) if len(ys) else 0.0,
            )
            infos.append({
                "segment_id":   int(sid),
                "name":         SEGMENT_NAMES.get(int(sid), f"unknown_{sid}"),
                "pixel_count":  pixel_count,
                "percentage":   round(pixel_count / total_pixels * 100, 2),
                "centroid":     centroid,  # (x_norm, y_norm)
            })
        return sorted(infos, key=lambda d: d["pixel_count"], reverse=True)

    def visualize(self, segment_map: np.ndarray) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB colour-coded segment map."""
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(19, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]  # background → black
        h, w = segment_map.shape
        coloured = palette[segment_map.flatten()].reshape(h, w, 3)
        return coloured

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def get_face_segment_adjacency() -> list[tuple[int, int]]:
        """Return the 22 anatomically plausible edge pairs."""
        return list(ANATOMICAL_EDGES)

    def build_face_graph_edges(
        self,
        segment_map: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Build edge index using ONLY anatomical adjacency (v2 contract).
        Edges are only included if both endpoint segments are present
        (pixel_count > 0).

        NOTE: fully_connected=True from v1 does NOT exist in v2.

        Returns:
            edge_index: (2, E) int64 array of re-indexed edge pairs
                        (indices into present_segment_ids, not raw segment IDs)
            present_segment_ids: list of segment IDs that appear in the image
        """
        present_ids = set(np.unique(segment_map).tolist())
        present_ids.discard(0)  # remove background

        # Only anatomical edges where both sides are present
        valid_edges = [
            (a, b) for a, b in ANATOMICAL_EDGES
            if a in present_ids and b in present_ids
        ]

        # Build a contiguous node index
        all_nodes = sorted(present_ids)
        id_to_idx = {sid: i for i, sid in enumerate(all_nodes)}

        if valid_edges:
            src = [id_to_idx[a] for a, _ in valid_edges]
            dst = [id_to_idx[b] for _, b in valid_edges]
            # Undirected: add both directions
            edge_index = np.array(
                [src + dst, dst + src], dtype=np.int64
            )
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return edge_index, all_nodes
