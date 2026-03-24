# Technical Reference — v1.0

> **Snapshot Date:** 2026-03-06  
> **Purpose:** Preserves every architectural decision, hyperparameter, and module contract for v1 so that future changes can be tracked against this baseline.

---

## 1. System Overview

**Goal:** Binary deepfake detection (Real / Fake) with human-readable forensic explanations.

**Pipeline (sequential per image):**

```
Input Image
    │
    ▼
┌─────────────┐      ┌───────────────┐
│ XceptionNet │      │   BiSeNet     │
│ (classifier)│      │ (face parser) │
└──────┬──────┘      └──────┬────────┘
       │                     │
  logits + layer        segment_map
  activations           (19 classes)
       │                     │
       ▼                     │
┌─────────────┐              │
│  LayerCAM   │              │
│ (attention) │              │
└──────┬──────┘              │
       │                     │
  attention_map              │
       │                     │
       ▼                     ▼
   ┌──────────────────────────┐
   │  Node Feature Extractor  │
   │  (per-segment features)  │
   └────────────┬─────────────┘
                │
         node_features
                │
                ▼
         ┌──────────┐
         │   GAT    │
         │ Explainer│
         └────┬─────┘
              │
     edge_importance
              │
              ▼
    ┌──────────────────┐
    │ Explanation Gen.  │
    │ (text + JSON)     │
    └──────────────────┘
```

---

## 2. Module Reference

### 2.1 XceptionNet Classifier

| Property | Value |
|---|---|
| **File** | `models/xception.py` |
| **Class** | `XceptionNetClassifier` |
| **Backbone** | `timm.create_model('xception', pretrained=True)` |
| **Output dim** | 2 (real=0, fake=1) |
| **Feature dim** | 2048 (after global avg pool) |
| **Input size** | 384 × 384 × 3 |
| **Dropout** | 0.5 (before final linear) |
| **Classifier init** | Xavier uniform weights, zero bias |
| **Normalization** | ImageNet — mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

**Hooked layers (for LayerCAM):**

| Hook name | XceptionNet block | Feature level |
|---|---|---|
| `block3` | Entry flow block 3 | Early — textures, edges |
| `block6` | Middle flow block 6 | Mid — patterns |
| `block12` | Exit flow block 12 | Late — semantics |

Both forward and backward hooks are registered for each.

**Key methods:**
- `forward(x, return_cam_features=False)` → logits or (logits, layer_outputs)
- `get_features(x)` → (B, 2048) pooled features
- `get_layer_features(x)` → dict of intermediate tensors
- `freeze_backbone(freeze_until='block6')` → freeze params up to named block
- `unfreeze_all()` → unfreeze everything
- `get_target_layers()` → list of nn.Module for CAM libraries
- `get_last_conv_layer()` → block12 or conv4

**Fallback:** `LightweightXception` class (no timm dependency, for testing only).

---

### 2.2 BiSeNet Face Parser

| Property | Value |
|---|---|
| **File** | `models/face_parser.py` |
| **Class** | `FaceParser` (wraps `BiSeNet`) |
| **Architecture** | SpatialPath + ContextPath (ResNet-like) + ARM + FFM |
| **Num classes** | 19 |
| **Input size** | 512 × 512 (internally resized) |
| **Output** | segment_map (H, W) with values 0–18 |

**19 Segment Labels (CelebAMask-HQ):**

| ID | Name | ID | Name |
|---|---|---|---|
| 0 | background | 10 | nose |
| 1 | skin | 11 | mouth |
| 2 | left_brow | 12 | upper_lip |
| 3 | right_brow | 13 | lower_lip |
| 4 | left_eye | 14 | neck |
| 5 | right_eye | 15 | necklace |
| 6 | eye_glasses | 16 | cloth |
| 7 | left_ear | 17 | hair |
| 8 | right_ear | 18 | hat |
| 9 | earring | | |

**Segment groups (for analysis):**

| Group | Segment IDs |
|---|---|
| eyes | 4, 5, 6 |
| brows | 2, 3 |
| nose | 10 |
| mouth | 11, 12, 13 |
| ears | 7, 8, 9 |
| skin | 1 |
| hair | 17, 18 |
| neck | 14, 15, 16 |

**BiSeNet sub-modules:**
- `ConvBNReLU` — Conv2d → BatchNorm2d → ReLU
- `SpatialPath` — 3× ConvBNReLU (stride 2 each), output: 256ch at 1/8 res
- `ContextPath` — ResNet-like (conv1 → 4 layers), two ARMs at 1/16 and 1/32
- `AttentionRefinementModule (ARM)` — conv + channel attention (sigmoid gate)
- `FeatureFusionModule (FFM)` — concat spatial+context → conv → channel attention
- `BiSeNet.forward(x)` → (B, 19, H, W) logits, upsampled to input size

**Key methods:**
- `parse(image)` → (H, W) uint8 segment map
- `get_segment_masks(segment_map)` → dict of binary masks
- `get_segment_info(segment_map)` → list of {segment_id, name, pixel_count, percentage, centroid}
- `visualize(segment_map)` → (H, W, 3) RGB colored image

**Graph construction:**
- `get_face_segment_adjacency()` → 22 anatomically plausible edge pairs
- `build_face_graph_edges(segment_map, fully_connected=True)` → (edge_index, present_segment_ids)
  - `fully_connected=True`: all-to-all edges between present segments
  - `fully_connected=False`: only anatomical adjacency edges

---

### 2.3 LayerCAM Attention

| Property | Value |
|---|---|
| **File** | `models/attention.py` |
| **Class** | `LayerCAMGenerator` |
| **Dependency** | `pytorch-grad-cam` library's `LayerCAM` |
| **Target layers** | block3, block6, block12 |
| **Fusion weights** | [0.2, 0.3, 0.5] (early→late) |
| **Output** | (H, W) float32 heatmap, values in [0, 1] |

**Fusion strategy:** Weighted sum of per-layer CAMs, each upsampled to `output_size` via bilinear interpolation, then min-max normalized.

**Also in this file:** `AttentionGenerator` class supporting `gradcam`, `gradcam++`, `scorecam`, `layercam` methods (for the original ResNet50 pipeline).

---

### 2.4 GAT Explainer

| Property | Value |
|---|---|
| **File** | `models/gat_explainer.py` |
| **Class** | `GATExplainer` |
| **Backend** | PyTorch Geometric `GATConv` (fallback: manual `GATLayer`) |
| **Layer 1** | GATConv(in → 128, heads=4, concat=True) → output: 512 |
| **Layer 2** | GATConv(512 → 128, heads=1, concat=False) → output: 128 |
| **Pooling** | `global_mean_pool` over all nodes → single graph vector (128,) |
| **Classifier** | Linear(128 → 64) → ReLU → Dropout(0.3) → Linear(64 → 2) |
| **Edge predictor** | Linear(256 → 128) → ReLU → Linear(128 → 1) → Sigmoid |
| **Dropout** | 0.3 |
| **BatchNorm** | After each GAT layer |

**Node feature vector (262 dimensions):**

| Feature | Dims | Source |
|---|---|---|
| Mean attention in segment | 1 | LayerCAM heatmap |
| Std attention | 1 | LayerCAM heatmap |
| Max attention | 1 | LayerCAM heatmap |
| Area ratio (segment/image) | 1 | Segment map |
| Pooled CNN features | 256 | XceptionNet backbone |
| Noise mean (Laplacian) | 1 | High-freq filter on image |
| Noise std (Laplacian) | 1 | High-freq filter on image |

**`NodeFeatureExtractor` (class):**
- `cnn_feature_dim` = 256
- `texture_feature_dim` = 2
- Total `feature_dim` = 4 + 256 + 2 = **262**
- Texture extraction: Grayscale → `scipy.ndimage.laplace()` → masked mean/std of absolute values

**Fallback `GATLayer`:** Manual multi-head attention with LeakyReLU(0.2), softmax per destination node. Used when `torch-geometric` is not installed.

**Batching:** `create_gat_batch()` offsets edge indices and creates batch assignment tensor. Uses PyG `Batch.from_data_list` if available.

---

### 2.5 Explanation Generator

| Property | Value |
|---|---|
| **File** | `utils/explanation.py` |
| **Functions** | `generate_explanation`, `generate_short_explanation`, `format_gat_explanation`, `generate_json_explanation` |

**Edge importance thresholds:**

| Score | Status | Template |
|---|---|---|
| > 0.7 | 🔴 SUSPICIOUS | "Blending boundary detected between {seg1} and {seg2}" |
| > 0.5 | 🟡 NOTABLE | "Texture inconsistency at {seg1} ↔ {seg2} junction" |
| ≤ 0.5 | 🟢 NORMAL | "Natural transition between {seg1} and {seg2}" |

**JSON output schema:**
```json
{
  "classification": "REAL|FAKE",
  "confidence": 0.0–1.0,
  "is_fake": true|false,
  "detected_segments": [{"name": "...", "percentage": 0.0}],
  "key_relationships": [
    {
      "segment_1": "...",
      "segment_2": "...",
      "importance_score": 0.0–1.0,
      "suspicious": true|false,
      "status": "suspicious|notable|normal"
    }
  ],
  "summary": "one-line summary"
}
```

---

### 2.6 Dataset Loaders

| Property | Value |
|---|---|
| **File** | `data/ff_dataset.py` |

**`FaceForensicsDataset`:**
- Expects FF++ directory structure with `original_sequences/`, `manipulated_sequences/`, `splits/`
- Returns `(image_tensor, label, metadata_dict)`
- `metadata_dict` = `{video_id, method, path}`
- `max_frames_per_video` = 10 (uniform sampling)
- Class balancing: downsample majority class

**`SimpleImageDataset`:**
- Expects `root/real/*.jpg` + `root/fake/*.jpg`
- Same return signature as above

**Transforms:**

| Split | Augmentation |
|---|---|
| train | Resize(384) → RandomHFlip(0.5) → ColorJitter(0.2,0.2,0.2,0.1) → ToTensor → Normalize |
| val/test | Resize(384) → ToTensor → Normalize |

---

## 3. Training Strategy

### Phase 1 — XceptionNet Pretraining

| Parameter | Value |
|---|---|
| Frozen layers | Everything up to `block6` |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Scheduler | CosineAnnealingWarmRestarts (T_0 = len(train_loader), T_mult = 2) |
| Loss | CrossEntropyLoss |
| Epochs | 5 |
| Checkpoint saved | `phase1_best.pt` (on best val acc) |

After Phase 1: all layers unfrozen.

### Phase 2 — Joint XceptionNet + GAT

| Parameter | Value |
|---|---|
| XceptionNet frozen layers | Everything up to `block3` |
| XceptionNet optimizer | AdamW, lr=1e-5 |
| GAT optimizer | AdamW, lr=1e-4 |
| Weight decay | 1e-5 |
| Lambda (consistency) | 0.1 |
| Loss | `xception_loss + 0.1 * gat_loss` |
| GAT images per batch | min(batch_size, 2) — memory limit |
| Epochs | 10 |
| Checkpoint saved | `phase2_best.pt` (xception + gat state dicts) |

**Per-image GAT step in Phase 2:**
1. Generate LayerCAM attention map (target_class=1, i.e. "fake")
2. Denormalize image → numpy uint8
3. Face parse → segment_map
4. Build fully-connected graph edges
5. Extract simple node features (mean_attn, max_attn, area_ratio, zero-padded to 262)
6. GAT forward → logits
7. CrossEntropyLoss(gat_logits, label)

---

## 4. Inference Pipeline

**File:** `inference.py` — class `DeepfakeInference`

**Steps:**
1. Resize to 384×384, normalize, forward through XceptionNet → logits, softmax
2. LayerCAM on the "fake" class (target_class=1) → attention_map at original resolution
3. BiSeNet parse → segment_map
4. Build fully-connected graph, extract node features (4 features: mean_attn, std_attn, max_attn, area_ratio; rest zero-padded)
5. GAT forward with `return_attention=True` → edge_importance
6. Generate text explanation + JSON
7. Matplotlib visualization: original / heatmap overlay / segment map / text summary

**CLI:**
```
python inference.py --image <path> --checkpoint <path> --output <dir> --device cuda --no-viz
```

---

## 5. File Inventory (v1)

| Path | Lines | Purpose |
|---|---|---|
| `models/xception.py` | 345 | XceptionNet classifier + hooks + fallback |
| `models/face_parser.py` | 533 | BiSeNet + FaceParser + graph builder |
| `models/attention.py` | 377 | AttentionGenerator + LayerCAMGenerator |
| `models/gat_explainer.py` | 565 | GATExplainer + NodeFeatureExtractor + batching |
| `models/__init__.py` | 25 | Package exports |
| `data/ff_dataset.py` | 424 | FF++ loader + SimpleImageDataset |
| `data/__init__.py` | 3 | Package init |
| `utils/explanation.py` | 234 | Text + JSON explanation generation |
| `utils/__init__.py` | 15 | Package exports |
| `train.py` | 616 | DeepfakeTrainer (Phase 1 + 2) |
| `inference.py` | 310 | DeepfakeInference + CLI |
| `requirements.txt` | 22 | Dependencies |
| `notebooks/kaggle_full_pipeline.py` | ~350 | Kaggle notebook (imports from dataset) |

---

## 6. Dependencies (v1)

| Package | Version | Used By |
|---|---|---|
| `torch` | ≥ 2.0 | Everything |
| `torchvision` | ≥ 0.15 | Transforms, data loading |
| `timm` | ≥ 0.9.0 | XceptionNet pretrained weights |
| `pytorch-grad-cam` | ≥ 1.5.0 | LayerCAM / GradCAM |
| `torch-geometric` | ≥ 2.3 | GATConv, global_mean_pool, Data, Batch |
| `scipy` | ≥ 1.10 | Laplacian filter for texture features |
| `opencv-python` | ≥ 4.8 | Image processing |
| `numpy` | ≥ 1.24 | Array operations |
| `pillow` | ≥ 10.0 | Image I/O |
| `matplotlib` | ≥ 3.7 | Visualization |
| `tqdm` | ≥ 4.65 | Progress bars |
| `pyyaml` | ≥ 6.0 | Config files |

---

## 7. Known Limitations (v1)

1. **BiSeNet weights are random** — no pretrained CelebAMask-HQ checkpoint is loaded by default. Segment maps will be noisy until a checkpoint is provided.
2. **Phase 2 GAT features are sparse** — only 3 of 262 feature dims are populated (mean_attn, max_attn, area_ratio). Full CNN + texture features are implemented in `NodeFeatureExtractor.extract()` but not called during training.
3. **No learning rate scheduler in Phase 2** — only Phase 1 uses CosineAnnealingWarmRestarts.
4. **Single-image GAT** — graphs are built per-image (not batched across the batch), limiting to 2 images per step for memory safety.
5. **No early stopping** — training runs for all configured epochs regardless of convergence.
6. **Demo data is random noise** — not real faces, so accuracy numbers from demo runs are meaningless. Intended only as a pipeline smoke-test.
