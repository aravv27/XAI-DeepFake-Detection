# Graph-Guided Explainable Deepfake Detection — v2

An explainable AI system that detects deepfakes and provides **causally-grounded** forensic reports. Combines facial segmentation, multi-scale attention heatmaps, and Graph Attention Networks (GAT) to explain *why* an image is considered fake based on inconsistencies between facial regions.

## What's New in v2

v2 is a ground-up rewrite fixing critical failures in v1. See [`v2_implementation_plan.md`](v2_implementation_plan.md) for full details.

| Area | v1 | v2 |
|---|---|---|
| GAT features | 3/262 dims populated | **10/10 dims fully computed** |
| BiSeNet | Random weights (silent) | **Pretrained checkpoint required** |
| Graph edges | Fully connected (noise) | **22 anatomical pairs only** |
| Explanations | Template thresholds | **GNNExplainer (causal)** |
| Evaluation | 99.8% on same distribution | **Cross-dataset: Celeb-DF v2 → DF40** |
| Training safeguards | None | **Gradient checks, early stopping, integration test** |

## Core Architecture

```
Input Image (384×384)
    ↓
XceptionNet   →  logits + layer activations
BiSeNet       →  segment_map (19 classes)        [pretrained weights REQUIRED]
LayerCAM      →  attention_map (multi-scale)
NodeExtractor →  node_features (N×10, ALL populated)
GAT           →  edge_importance
GNNExplainer  →  causal edge masks
Report Gen    →  .txt + .json + .png
```

## Installation

```bash
git clone <repository-url>
cd XAI
pip install -r requirements.txt
```

**Required:** Download [BiSeNet CelebAMask-HQ weights](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) and place as `checkpoints/bisenet.pth`.

## Running the Pipeline

### 1. Integration Verification (Required First)

Before any training, verify the full pipeline works end-to-end:

```bash
python verify_pipeline.py --image path/to/face.jpg --bisenet-checkpoint checkpoints/bisenet.pth
```

All 5 checks must pass before training can proceed.

### 2. Training (Kaggle GPU Recommended)

**Option A: Kaggle Notebook**
1. Upload this repo as a Kaggle dataset
2. Attach [Celeb-DF v2](https://www.kaggle.com/datasets/pranabr0y/celebdf-v2image-dataset) dataset
3. Upload BiSeNet checkpoint as a dataset
4. Run `notebooks/kaggle_training.py` with GPU T4 x2 enabled

**Option B: Local Training**
```bash
python train.py --mode full --data-root path/to/celebdf --bisenet-checkpoint checkpoints/bisenet.pth
```

**Training modes** (for ablation):
- `--mode xception_only` — baseline classifier, no graph
- `--mode full` — XceptionNet + GAT joint training
- `--mode gat_only` — sanity check; should underperform xception_only

### 3. Inference & Explanations

```bash
python inference.py \
    --image path/to/suspect.jpg \
    --checkpoint checkpoints/phase2_best.pt \
    --bisenet-checkpoint checkpoints/bisenet.pth \
    --output results/
```

## Understanding the Outputs

For every image, the system generates:

1. **`{image}_analysis.png`** — 4-panel dashboard: original, LayerCAM heatmap, segmentation map, analysis summary
2. **`{image}_explanation.txt`** — Human-readable forensic report with causal edge importance
3. **`{image}_result.json`** — Structured data with GNNExplainer edge masks, segment info, and classification confidence

## Project Structure

```
XAI/
├── data/
│   └── dataset.py             # Celeb-DF v2 + DF40 DataLoaders
├── models/
│   ├── xception.py            # XceptionNet classifier + LayerCAM hooks
│   ├── attention.py           # LayerCAM multi-scale attention
│   ├── face_parser.py         # BiSeNet segmentation + anatomical graph
│   └── gat_explainer.py       # GAT + NodeFeatureExtractor (10 dims)
├── utils/
│   ├── explanation.py         # GNNExplainer + report generation
│   └── visualization.py       # 4-panel dashboard rendering
├── notebooks/
│   └── kaggle_training.py     # Ready-to-run Kaggle training script
├── verify_pipeline.py         # Integration test (must pass before training)
├── train.py                   # Two-phase trainer with gradient checks
├── inference.py               # End-to-end inference + CLI
├── requirements.txt           # Hard dependencies
├── techRef.md                 # v1 technical reference (baseline)
└── v2_implementation_plan.md  # v2 architecture contracts
```

## Datasets

| Dataset | Purpose | Source |
|---|---|---|
| Celeb-DF v2 | Training + validation | [Kaggle](https://www.kaggle.com/datasets/pranabr0y/celebdf-v2image-dataset) |
| DF40 | Cross-dataset evaluation (40 methods) | [HuggingFace](https://huggingface.co/datasets/aibio-aotearoa/DF40_test_subset) |

## Evaluation Metrics

- **AUC-ROC** — primary detection metric
- **F1 Score** — balanced comparison metric
- **Cross-dataset gap** — (Celeb-DF AUC) − (DF40 AUC) — key research finding
