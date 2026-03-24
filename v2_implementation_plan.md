# Graph-Guided Explainable Deepfake Detection — v2 Implementation Plan

> **Date:** March 2026  
> **Type:** Research Project  
> **Companion to:** `README.md` and `techRef.md` (v1.0)  
> **Purpose:** Complete ground-up rewrite. Every architectural decision, dataset contract, training rule, and v1 failure mitigation is defined here. Pass this document alongside README.md and techRef.md to code generation. Everything in this document is a hard contract, not a suggestion.

---

## 1. Why a Rewrite

v1 was built module-by-module without end-to-end integration testing. Three confirmed failures made the system non-functional as an XAI tool:

- **GAT loss = 0.0000 for all 10 Phase 2 epochs** — the graph module learned nothing
- **259 of 262 node feature dimensions were zero** — NodeFeatureExtractor.extract() was implemented but never called
- **BiSeNet used random weights silently** — no pretrained checkpoint was loaded, so all segmentations were noise

The system produced confident-sounding forensic explanations derived entirely from random numbers. The XceptionNet classifier worked. Nothing else did.

v2 fixes this by enforcing integration contracts at every module boundary before training begins.

---

## 2. v1 Failure Analysis & v2 Mitigations

| v1 Failure | Root Cause | v2 Mitigation |
|---|---|---|
| GAT loss = 0.0000 all epochs | Gradient not flowing to GAT; sparse features gave near-zero loss signal | Assert GAT grad norm > 0 before Phase 2; verified in integration test |
| 259/262 node feature dims zeroed | NodeFeatureExtractor.extract() never called in training loop | All 10 feature dims fully populated from day one; no optional dims |
| BiSeNet produces random segmentations | No pretrained checkpoint loaded; silent fallback to random weights | Pretrained CelebAMask-HQ weights are hard dependency; FileNotFoundError raised if missing |
| Edge importance scores all ~50% | Consequence of random BiSeNet + zero GAT gradients | Resolved by above two mitigations |
| Explanations fabricated from noise | Template engine applied to meaningless scores | GNNExplainer used for causally-grounded edge attribution |
| 99.8% accuracy is meaningless | Trained and evaluated on same StyleGAN distribution | Train on Celeb-DF v2; evaluate on DF40 (40 different generation methods) |
| Fully-connected graph dilutes signal | All-to-all edges including hat↔lower_lip | Anatomical edges only (22 pairs) in v2 |

---

## 3. Dataset Strategy

### 3.1 Training Dataset — Celeb-DF v2

**Source:** Kaggle image extract (`pranabr0y/celebdf-v2image-dataset`). No approval required, available immediately.

| Property | Value |
|---|---|
| Real videos (source) | 590 YouTube celebrity clips |
| Fake videos | 5,639 high-quality DeepFake videos |
| Pre-extracted frames | Yes — JPEG frames ready for DataLoader |
| Diversity | Multiple ethnicities, ages, genders |
| Why chosen | Standard cross-dataset benchmark; harder than 140k StyleGAN dataset |

### 3.2 Evaluation Dataset — DF40 Test Subset

**Source:** HuggingFace (`aibio-aotearoa/DF40_test_subset`). CC BY-NC 4.0. No approval required.

| Property | Value |
|---|---|
| Real images | 16,060 |
| Fake images | 16,060 |
| Generation methods covered | 40 distinct techniques |
| Includes | HeyGen, MidJourney, DeepFaceLab, diffusion-based swaps, face reenactment |
| Why chosen | Hardest modern benchmark; tests generalisation to unseen generation methods |

### 3.3 The Experimental Claim

Training on Celeb-DF v2 and evaluating on DF40 is a stronger setup than the FF++ → Celeb-DF split used by most existing work, because DF40 includes contemporary tools (HeyGen, MidJourney) that post-date most published baselines. The gap between in-distribution and cross-dataset accuracy is the primary research contribution metric.

### 3.4 Ablation Datasets

Three model variants will be trained and evaluated identically:

- `XceptionNet alone` — baseline classifier, no graph
- `Full system` — XceptionNet + GAT with all features
- `GAT alone` — sanity check; should underperform XceptionNet alone

If the full system does not outperform XceptionNet alone on DF40, the GAT module is not earning its place and the architecture must be reconsidered before claiming explainability as a contribution.

---

## 4. v2 Architecture

### 4.1 Pipeline

```
Input Image (384×384×3)
    ↓
XceptionNet   →  logits + layer activations
BiSeNet       →  segment_map (H×W, values 0–18)   [pretrained weights REQUIRED]
LayerCAM      →  attention_map (H×W, float [0,1])
NodeExtractor →  node_features (N×10, ALL dims populated)
GAT           →  edge_importance (E×1)
GNNExplainer  →  causal edge masks
Report Gen    →  .txt + .json + .png
```

### 4.2 XceptionNet Classifier

| Property | Value |
|---|---|
| File | `models/xception.py` |
| Backbone | `timm.create_model('xception', pretrained=True)` |
| Input size | 384 × 384 × 3 |
| Output | 2-class logits (real=0, fake=1) |
| Feature dim | 2048 after global avg pool |
| Dropout | 0.5 before final linear |
| CAM hook layers | block3 (textures), block6 (patterns), block12 (semantics) |
| Normalisation | ImageNet mean/std |
| Change from v1 | None — this module was correct in v1 |

### 4.3 BiSeNet Face Parser

| Property | Value |
|---|---|
| File | `models/face_parser.py` |
| Pretrained weights | CelebAMask-HQ checkpoint — **HARD DEPENDENCY** |
| Startup behaviour | Raise `FileNotFoundError` with download URL if checkpoint missing |
| Startup validation | Run forward pass on blank tensor; assert output shape (1, 19, H, W); if all pixels are class 0 raise RuntimeError |
| Output | segment_map (H×W) uint8, values 0–18 |
| Num classes | 19 (unchanged from v1) |
| Change from v1 | Hard dependency enforcement; **no silent random-weight fallback under any circumstance** |

### 4.4 LayerCAM Attention

| Property | Value |
|---|---|
| File | `models/attention.py` |
| Library | `pytorch-grad-cam` LayerCAM |
| Target layers | block3, block6, block12 |
| Fusion weights | [0.2, 0.3, 0.5] early→late |
| Output | (H×W) float32 heatmap, values in [0, 1] |
| Change from v1 | None — this module was correct in v1 |

### 4.5 Node Feature Extractor — v2 Contract

**This is the most critical change from v1. All 10 dimensions are fully populated for every node. There are no zero-padded placeholder dims. Zero-padding to reach a target dim is forbidden.**

| Dim | Feature | Source | In v1? |
|---|---|---|---|
| 1 | Mean LayerCAM attention in segment | LayerCAM heatmap | Yes |
| 2 | Std LayerCAM attention in segment | LayerCAM heatmap | Yes |
| 3 | Max LayerCAM attention in segment | LayerCAM heatmap | Yes |
| 4 | Area ratio (segment pixels / image pixels) | Segment map | Yes |
| 5 | Laplacian noise mean (texture frequency) | scipy.ndimage.laplace on greyscale | No — fixed in v2 |
| 6 | Laplacian noise std | scipy.ndimage.laplace on greyscale | No — fixed in v2 |
| 7 | Mean R channel in segment | Raw image pixels | No — new in v2 |
| 8 | Mean G channel in segment | Raw image pixels | No — new in v2 |
| 9 | Centroid X (normalised 0–1) | Segment map | No — new in v2 |
| 10 | Centroid Y (normalised 0–1) | Segment map | No — new in v2 |

**Note:** CNN-pooled features (256 dims from v1) are deliberately excluded from v2. Prove the GAT learns on clean geometric and texture features first. CNN features can be added in a later experiment once baseline GAT gradient is confirmed.

### 4.6 Graph Construction

| Property | Value |
|---|---|
| Edge type | Anatomical adjacency only — 22 predefined pairs |
| Fully-connected mode | **Removed entirely.** Was the default in v1, produced meaningless edges |
| Edge filtering | Only include edges where both segments are present (pixel count > 0) |
| Why anatomical only | hat↔lower_lip carries no forensic meaning. Constraining to anatomy forces the GAT to learn meaningful spatial inconsistency |

### 4.7 GAT Explainer

| Property | Value |
|---|---|
| File | `models/gat_explainer.py` |
| Backend | PyTorch Geometric GATConv — **no manual fallback in v2** |
| Input dim | 10 (fully populated — see 4.5) |
| Layer 1 | GATConv(10 → 64, heads=4, concat=True) → output: 256 |
| Layer 2 | GATConv(256 → 64, heads=1, concat=False) → output: 64 |
| Pooling | global_mean_pool → graph vector (64,) |
| Classifier | Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→2) |
| Dropout | 0.3 |
| BatchNorm | After each GAT layer |
| Change from v1 | Input dim reduced from 262→10 (all populated). Architecture simplified to match feature count. |

### 4.8 Explanation Engine

| Property | Value |
|---|---|
| File | `utils/explanation.py` |
| Method | GNNExplainer (PyG built-in) — **replaces threshold templates entirely** |
| What it produces | Edge masks causally tied to classification decision |
| Why | Raw GAT attention weights ≠ importance. GNNExplainer masks reflect actual decision contribution. |
| Output formats | `.txt` forensic report, `.json` structured data, `.png` 4-panel dashboard |
| Template fallback | **None.** If GNNExplainer fails, report is flagged incomplete — never fabricated from thresholds |

---

## 5. Integration Contract

These rules exist because v1 violated them. Every seam between modules must be verified before training begins.

### 5.1 Pre-Training Integration Test (`verify_pipeline.py`)

Before any training epoch runs, execute `verify_pipeline.py` with a single real face image. This script must assert all of the following — if any assertion fails, training does not start:

- BiSeNet produces a non-uniform segment map (at least 3 distinct segment IDs present)
- Node feature matrix has shape (N, 10) with no NaN and no all-zero rows
- GAT forward pass produces logits with shape (1, 2)
- A single backward pass produces GAT parameter gradients with norm > 1e-6
- Inference pipeline produces a `.json` file with `key_relationships` populated

### 5.2 Phase 2 Gradient Check

At the end of every Phase 2 epoch, log:

- GAT parameter gradient norm
- GAT loss value (must be > 0 and changing between epochs)
- Number of graphs processed (must equal number of images in batch)

If GAT grad norm is zero for two consecutive epochs, raise `RuntimeError` and halt. Do not silently continue with a dead module.

### 5.3 BiSeNet Startup Check

`FaceParser.__init__()` must:

- Accept `checkpoint_path` as a required argument with no default value
- Raise `FileNotFoundError` with message including the CelebAMask-HQ download URL if file not found
- After loading, run a forward pass on a blank tensor and assert output shape is (1, 19, H, W)
- Log segment distribution on a test image — if all pixels are class 0 (background), raise `RuntimeError`

---

## 6. Training Strategy

### 6.1 Phase 1 — XceptionNet Pretraining

| Parameter | Value |
|---|---|
| Dataset | Celeb-DF v2 (train split) |
| Frozen layers | Everything up to block6 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Scheduler | CosineAnnealingWarmRestarts (T_0 = len(train_loader), T_mult = 2) |
| Loss | CrossEntropyLoss |
| Epochs | 5 |
| Early stopping | Patience = 3 epochs on val loss (new in v2) |
| Checkpoint | `phase1_best.pt` on best val acc |
| After Phase 1 | Unfreeze all layers before Phase 2 |

### 6.2 Phase 2 — Joint XceptionNet + GAT

| Parameter | Value |
|---|---|
| XceptionNet frozen layers | Everything up to block3 |
| XceptionNet optimizer | AdamW, lr=1e-5 |
| GAT optimizer | AdamW, lr=1e-4 |
| Weight decay | 1e-5 |
| LR scheduler | CosineAnnealingLR on both optimizers (new in v2 — was missing from v1) |
| Lambda (consistency) | 0.1 → annealed to 0.5 over 10 epochs (new in v2) |
| Loss | `xception_loss + lambda * gat_loss` |
| GAT images per step | All images in batch (not limited to 2 as in v1) |
| Epochs | 10 |
| Early stopping | Patience = 3 epochs on val loss |
| Checkpoint | `phase2_best.pt` (xception + gat state dicts) |

**Lambda annealing rationale:** Starting at 0.1 lets XceptionNet stabilise first. Increasing to 0.5 by epoch 10 forces the GAT to contribute meaningfully to the joint loss. This is the key structural fix from v1, where the fixed 0.1 weight combined with near-zero features produced zero gradient signal throughout Phase 2.

### 6.3 Ablation Training

Train all three variants on the same Celeb-DF v2 split with identical hyperparameters:

```
python train.py --mode xception_only
python train.py --mode full
python train.py --mode gat_only
```

Evaluate all three on DF40 test subset. Report accuracy, AUC-ROC, and F1 for each. This ablation table is the primary evidence for the GAT module's contribution.

---

## 7. v2 File Structure

| Path | Purpose | Change from v1 |
|---|---|---|
| `models/xception.py` | XceptionNet classifier + LayerCAM hooks | Unchanged |
| `models/face_parser.py` | BiSeNet + FaceParser (hard checkpoint dependency) | Startup validation; no silent fallback |
| `models/attention.py` | LayerCAMGenerator | Unchanged |
| `models/gat_explainer.py` | GATExplainer (input_dim=10) + NodeFeatureExtractor | All 10 dims populated; simplified architecture |
| `models/__init__.py` | Package exports | Minor |
| `data/dataset.py` | CelebDF v2 + DF40 DataLoaders | New file; replaces ff_dataset.py |
| `utils/explanation.py` | GNNExplainer + report generation | GNNExplainer replaces template engine |
| `utils/visualization.py` | Heatmap + graph rendering | Minor updates |
| `utils/__init__.py` | Package exports | Minor |
| `verify_pipeline.py` | End-to-end integration test | **New file — did not exist in v1** |
| `train.py` | Two-phase trainer with --mode flag and gradient checks | Gradient assertions; LR scheduler Phase 2; early stopping |
| `inference.py` | End-to-end inference + CLI | GNNExplainer output; verified feature extraction |
| `requirements.txt` | Dependencies | torch-geometric is hard requirement (no optional fallback) |
| `notebooks/kaggle_training.ipynb` | Kaggle GPU notebook | Updated for Celeb-DF v2 + DF40 |

---

## 8. Dependencies

All dependencies are hard requirements. Optional fallbacks were a root cause of v1's silent failures.

| Package | Version | Notes |
|---|---|---|
| `torch` | ≥ 2.0 | |
| `torchvision` | ≥ 0.15 | |
| `timm` | ≥ 0.9.0 | XceptionNet pretrained weights |
| `pytorch-grad-cam` | ≥ 1.5.0 | LayerCAM |
| `torch-geometric` | ≥ 2.3 | No manual fallback in v2 |
| `scipy` | ≥ 1.10 | Laplacian texture features |
| `opencv-python` | ≥ 4.8 | |
| `numpy` | ≥ 1.24 | |
| `pillow` | ≥ 10.0 | |
| `matplotlib` | ≥ 3.7 | |
| `tqdm` | ≥ 4.65 | |
| `pyyaml` | ≥ 6.0 | |
| `scikit-learn` | ≥ 1.3 | AUC/F1 metrics for ablation |

---

## 9. Evaluation Protocol

### 9.1 Metrics

- **AUC-ROC** — primary metric for binary detection
- **F1 Score** — primary metric for balanced comparison
- **Accuracy** — reported for context only
- **Cross-dataset gap** — (in-distribution AUC) minus (DF40 AUC) — the key research finding

### 9.2 Evaluation Splits

| Split | Dataset | Purpose |
|---|---|---|
| Train | Celeb-DF v2 (80%) | Model training |
| Validation | Celeb-DF v2 (20%) | Hyperparameter tuning + early stopping |
| Test in-distribution | Celeb-DF v2 held-out | In-distribution performance |
| Test cross-dataset | DF40 full test subset | Generalisation — primary claim |

### 9.3 Explainability Faithfulness Test

Accuracy alone does not validate the XAI claim. Run this test before claiming explainability as a contribution:

1. Run GNNExplainer on 100 correctly-classified fake images
2. Mask the top-3 suspicious edges identified by GNNExplainer and re-classify
3. If accuracy drops >10%, the explanations are causally linked to the decision — report this
4. If accuracy is unchanged, GNNExplainer masks are not faithful — **report this honestly, do not hide it**

---

## 10. Explicit Constraints for Code Generation

The following patterns caused v1 failures. They must not appear anywhere in v2 code.

### NEVER: Optional or silent fallbacks
- If BiSeNet weights are missing → raise `FileNotFoundError`, do not use random weights
- If torch-geometric is missing → raise `ImportError`, do not use a manual GATLayer
- There is no graceful degradation mode. Either the full system runs or it errors clearly.

### NEVER: Placeholder feature dimensions
- Every column of the node feature matrix must contain a real computed value
- Zero-padding to reach a target dimensionality is forbidden
- If a feature cannot be computed for a segment, that segment is excluded from the graph

### NEVER: Training without gradient verification
- Phase 2 must assert GAT parameter grad norm > 1e-6 after the first backward pass
- If grad norm is zero, raise `RuntimeError` — do not proceed

### NEVER: Template-based explanations
- Thresholding a raw attention weight and printing a fixed string is not XAI
- Use GNNExplainer exclusively for edge attribution
- The explanation module has no threshold-to-template logic anywhere

### NEVER: Fully-connected graphs
- Only anatomically plausible edges (22 pairs defined in `face_parser.py`)
- The `fully_connected=True` parameter from v1 does not exist in v2

### NEVER: Evaluation on training distribution only
- DF40 cross-dataset evaluation is mandatory before any accuracy claim is stated
- In-distribution accuracy (Celeb-DF v2 test) is reported but never used as the primary result

---

## 11. Summary — v1 vs v2

| Dimension | v1 | v2 |
|---|---|---|
| GAT contribution | Zero (confirmed by training logs) | Verified by gradient check before training |
| Node features | 3/262 dims populated | 10/10 dims populated |
| BiSeNet | Random weights (silent) | Pretrained weights (hard dependency) |
| Graph edges | Fully connected (noise) | Anatomical only (22 pairs) |
| Explanations | Fixed templates on meaningless scores | GNNExplainer with faithfulness test |
| Evaluation | 99.8% on StyleGAN-only dataset | AUC/F1 on Celeb-DF v2 + DF40 cross-dataset |
| Phase 2 LR scheduler | Missing | CosineAnnealingLR on both optimizers |
| Lambda | Fixed 0.1 (too low) | Annealed 0.1 → 0.5 over 10 epochs |
| Integration test | None | `verify_pipeline.py` runs before any training |
| Early stopping | None | Patience = 3 on val loss, both phases |

---

*Pass this document alongside `README.md` and `techRef.md` (v1.0) to code generation.*  
*Everything above is a hard contract, not a suggestion.*
