# Graph-Guided Explainable Deepfake Detection

An advanced explainable AI system that detects deepfakes and provides human-readable forensic reports. It combines facial segmentation, attention heatmaps, and Graph Attention Networks (GAT) to explain *why* an image is considered fake based on the relationships between facial regions.

## Core Architecture

This system has been upgraded from a generic classifier to a specialized facial forensic tool:

1. **Classification Backbone**: XceptionNet (Binary: Real vs. Fake)
2. **Feature Localization**: LayerCAM for multi-scale attention (Textures + Semantics)
3. **Semantic Parsing**: BiSeNet for extracting 19 distinct facial segments (eyes, nose, skin, boundaries)
4. **Relationship Modeling**: Graph Attention Networks (GAT) to model the consistency between adjacent facial segments.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd XAI

# Install dependencies (requires PyTorch and PyTorch Geometric)
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Training (Kaggle Recommended)
To train the model from scratch on the FaceForensics++ dataset, we highly recommend using a GPU environment like Kaggle.

1. Create a new Kaggle Notebook (with GPU T4 x2 enabled).
2. Upload the `notebooks/kaggle_training.ipynb` file (or copy the contents of `notebooks/kaggle_training.py` into a Script).
3. Connect the FaceForensics++ dataset to your Kaggle environment.
4. Run all cells to execute the two-phase training (XceptionNet Pretraining -> Joint GAT Fine-tuning).

### 2. Inference & Explanations

To run the model on a single image and generate a forensic report:

```bash
python inference.py --image path/to/suspect_image.jpg --output results/
```

**Options:**
- `--image, -i`: Path to the input image (Required)
- `--checkpoint, -c`: Path to your trained `.pt` weights
- `--output, -o`: Directory to save visual results and JSON reports
- `--device, -d`: strictly use `cuda` or `cpu` (default: auto-detect)
- `--no-viz`: Run headless without displaying the matplotlib popup

## Understanding the Outputs

For every image processed through `inference.py`, the system generates:

1. **`{image}_analysis.png`**: A 2x2 visual dashboard containing:
   - The original image
   - The LayerCAM attention heatmap (where the model is looking)
   - The BiSeNet face segmentation map
   - A text summary of the top suspicious facial relationships
2. **`{image}_explanation.txt`**: A detailed, human-readable forensic report explaining the model's reasoning (e.g., *"Texture inconsistency at skin ↔ left_eye junction"*). 
3. **`{image}_result.json`**: A structured JSON file containing the raw confidence scores, segmentation breakdowns, and GAT edge importance metrics for API integration.

## Project Structure

```
XAI/
├── data/
│   └── ff_dataset.py      # FaceForensics++ data loaders & augmentation
├── models/
│   ├── xception.py        # Backbone Classifier
│   ├── attention.py       # LayerCAM Multi-scale Attention
│   ├── face_parser.py     # BiSeNet Segmentation
│   └── gat_explainer.py   # Graph Attention Network
├── utils/
│   ├── explanation.py     # NLG rule-engine for forensic reports
│   └── visualization.py   # Heatmap and Graph rendering
├── notebooks/
│   └── kaggle_training.ipynb # Ready-to-run GPU training notebook
├── train.py               # Local two-phase training script
├── inference.py           # End-to-end inference & explanation CLI
└── requirements.txt
```

## Future Enhancements (Phase 2)
See `implementation_plan_v2.md` for our upcoming roadmap, which includes Temporal Video Analysis (ST-GAT) and an interactive Streamlit forensic dashboard.
