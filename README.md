# XAI Classification + Segmentation Alignment System

An explainable AI system that combines image classification attention maps with semantic segmentation to create interpretable visualizations for non-technical users.

## Features

- **Image Classification**: Pre-trained ResNet50 on ImageNet (1000 classes)
- **Attention Maps**: Grad-CAM, Grad-CAM++, and Score-CAM support
- **Semantic Segmentation**: DeepLabV3+ on COCO (21 classes)
- **Alignment Metrics**: IoU, coverage, precision, recall
- **Visualizations**: Heatmap overlays, segmentation maps, comparison grids
- **Human-Readable Explanations**: Natural language interpretation of model decisions

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic usage
python main.py --image path/to/image.jpg --output output/

# With Score-CAM (more accurate but slower)
python main.py --image photo.jpg --method scorecam

# Custom threshold
python main.py --image photo.jpg --threshold 0.3
```

## Output Files

For each input image, the system generates:
- `{name}_heatmap.png` - Attention heatmap overlay
- `{name}_segmentation.png` - Semantic segmentation overlay
- `{name}_combined.png` - Combined visualization
- `{name}_grid.png` - 2x2 comparison grid
- `{name}_metrics.json` - Quantitative metrics
- `{name}_explanation.txt` - Human-readable explanation

## Project Structure

```
XAI/
├── models/
│   ├── classifier.py      # ResNet50 classification
│   ├── attention.py       # Grad-CAM/Score-CAM
│   └── segmentation.py    # DeepLabV3+ segmentation
├── utils/
│   ├── preprocessing.py   # Image preprocessing
│   ├── labels.py          # Class name mappings
│   ├── alignment.py       # IoU and metrics
│   └── visualization.py   # Overlay generation
├── configs/
│   └── config.yaml        # Settings
├── examples/
│   └── sample_images/     # Test images
├── main.py                # CLI interface
└── requirements.txt
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--image, -i` | Input image path | Required |
| `--output, -o` | Output directory | `output/` |
| `--method, -m` | Attention method | `gradcam` |
| `--threshold, -t` | Attention threshold | `0.2` |
| `--device, -d` | Device (cuda/cpu) | Auto |

## Metrics Explained

- **IoU**: Overlap between attention region and segmented object
- **Coverage**: Percentage of attention falling within the correct segment
- **Precision**: How much of the attended region is correct
- **Recall**: How much of the object is attended to
