# XAI Classification + Segmentation Alignment Project

## Project Overview

This project combines **image classification with explainable AI (XAI)** using attention heatmaps (Grad-CAM variants) and **semantic segmentation** to create interpretable visualizations for non-technical stakeholders. The system shows where a classification model focuses its attention and maps these regions to human-readable semantic labels.

### Core Objective
Generate visual explanations showing:
1. Original image
2. Classification prediction with confidence
3. Grad-CAM/Score-CAM attention heatmap (where model focuses)
4. Semantic segmentation overlay with labeled regions
5. Quantitative alignment metrics (IoU between attention and segments)
6. Human-readable explanation: "Model classified this as 'dog' with 94% confidence, focusing 78% attention on the face region and 15% on the ears"

---

## Architecture Components

### 1. Classification Model + Attention Maps
- **Purpose**: Generate class predictions and attention heatmaps
- **Models**: Pre-trained ResNet50, EfficientNet-B0, or ViT (torchvision/timm)
- **XAI Method**: Score-CAM (preferred, IoU 0.34) or Grad-CAM++ (better than vanilla Grad-CAM)
- **Research**: Score-CAM achieves superior localization (IoU 0.34 vs 0.19 for Grad-CAM) [ArXiv 2510.24414v1]
- **Library**: `pytorch-grad-cam` by jacobgil

### 2. Semantic Segmentation Model
- **Purpose**: Generate pixel-wise semantic labels for image regions
- **Models**: 
  - DeepLabV3+ (ResNet101 backbone) - strong general performance
  - Mask R-CNN - instance-aware segmentation
  - Segment Anything Model (SAM) - zero-shot capabilities
- **Datasets**: Pre-trained on COCO (80 classes) or ADE20K (150 classes)
- **Library**: `torchvision.models.segmentation` or `mmsegmentation`

### 3. Alignment & Metrics
- **Purpose**: Quantify how well attention maps align with segmentation masks
- **Metrics**:
  - **IoU (Intersection over Union)**: Overlap between attention region and segment
  - **Precision/Recall**: Attention focus within correct segments
  - **Coverage %**: Percentage of attention on predicted class segment
- **Research**: Use thresholding (top 20-30% attention values) before IoU calculation

### 4. Visualization Pipeline
- **Purpose**: Create overlays and interpretable outputs
- **Outputs**:
  - Attention heatmap overlay (semi-transparent, jet/hot colormap)
  - Segmentation boundaries with class labels
  - Combined visualization with all layers
  - Side-by-side comparison
- **Tools**: OpenCV, Matplotlib, PIL

### 5. Interactive Dashboard
- **Purpose**: Allow non-technical users to upload images and explore
- **Framework**: Streamlit (preferred for speed) or Gradio
- **Features**:
  - Image upload
  - Model selection dropdown
  - Real-time inference
  - Interactive sliders (heatmap opacity, threshold)
  - Downloadable reports

---

## Implementation Steps

### Step 1: Environment Setup
```yaml
Dependencies:
  - torch >= 2.0
  - torchvision >= 0.15
  - grad-cam >= 1.5.0
  - opencv-python >= 4.8
  - matplotlib >= 3.7
  - numpy >= 1.24
  - pillow >= 10.0
  - streamlit >= 1.30 (optional, for dashboard)
  - timm >= 0.9 (optional, for more models)
```

### Step 2: Load Pre-trained Models
- Classification: Load ImageNet pre-trained ResNet50
- Segmentation: Load COCO pre-trained DeepLabV3+
- No fine-tuning needed - use frozen weights
- Both models work on CPU (slower) or GPU

### Step 3: Classification + Attention Generation
```python
Steps:
1. Preprocess image (resize 224x224, normalize)
2. Forward pass through classifier
3. Get top-k predictions
4. Generate Score-CAM/Grad-CAM++ heatmap for top prediction
5. Upsample heatmap to original image size
6. Apply threshold to get binary attention mask
```

**Research Note**: Score-CAM removes gradient dependence, making it more reliable than Grad-CAM for fine-grained localization.

### Step 4: Semantic Segmentation
```python
Steps:
1. Preprocess image (segmentation models often use different sizes)
2. Forward pass through segmentation model
3. Get per-pixel class predictions
4. Map class IDs to human-readable labels (COCO classes)
5. Generate colored segmentation mask
6. Extract boundaries for overlay
```

**Label Mapping Example**:
- Class 0 → "Background"
- Class 1 → "Person"
- Class 18 → "Dog"
- Class 63 → "Laptop"

### Step 5: Alignment Analysis
```python
Steps:
1. Threshold attention heatmap (top 20% → binary mask)
2. Extract segmentation mask for predicted class only
3. Calculate IoU = intersection(attention, segment) / union(attention, segment)
4. Calculate coverage = sum(attention * segment) / sum(attention)
5. Generate metrics dictionary
```

**Threshold Selection**: Research shows 20-30% threshold on attention values provides best alignment with human perception.

### Step 6: Visualization Generation
```python
Layers to create:
1. Base image
2. Attention heatmap (alpha=0.4, colormap='jet')
3. Segmentation boundaries (thick colored lines)
4. Text labels for segments (class name + confidence)
5. Metrics overlay (IoU, coverage %)

Output formats:
- Combined overlay image
- 2x2 grid (original, attention, segmentation, combined)
- Interactive HTML with hover tooltips
```

### Step 7: Natural Language Explanation
```python
Template:
"The model classified this image as '{class_name}' with {confidence}% confidence. 
The model focused {coverage}% of its attention on the {segment_label} region, 
which achieved {iou}% alignment with the actual {segment_label} boundaries."

Example:
"The model classified this image as 'Golden Retriever' with 94% confidence. 
The model focused 78% of its attention on the dog's face region, 
which achieved 71% alignment with the actual face boundaries."
```

---

## Code Structure

```
project/
├── models/
│   ├── classifier.py          # Load and run classification models
│   ├── segmentation.py        # Load and run segmentation models
│   └── attention.py           # Generate Grad-CAM/Score-CAM
├── utils/
│   ├── preprocessing.py       # Image preprocessing pipelines
│   ├── postprocessing.py      # Heatmap/mask processing
│   ├── alignment.py           # IoU and metrics calculation
│   ├── visualization.py       # Overlay generation
│   └── labels.py              # Class ID to name mapping (COCO/ImageNet)
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
├── configs/
│   └── config.yaml            # Model paths, thresholds, settings
├── examples/
│   └── example_images/        # Test images
├── notebooks/
│   └── experiments.ipynb      # Research and testing
├── requirements.txt
├── README.md
└── main.py                    # CLI interface
```

---

## Key Research References

### XAI Methods
1. **Score-CAM** (Wang et al.): Removes gradient dependency, achieves IoU 0.34 vs Grad-CAM 0.19
2. **Grad-CAM++** (Chattopadhay et al.): Better localization for multiple instances
3. **XAI Evaluation Framework** (ArXiv 2510.24414v1): Metrics for comparing attention with segmentation

### Segmentation
1. **DeepLabV3+**: Atrous convolution for multi-scale segmentation
2. **Segment Anything Model (SAM)**: Zero-shot segmentation, highly flexible
3. **COCO Dataset**: 80 object categories with instance annotations

### Interpretability
1. **Enhanced SegNet with Grad-CAM** (PMC integration): Achieves 95.77% accuracy with visual explanations
2. **Explainable AI Dashboards** (TowardsAI): Best practices for non-technical user interfaces

---

## Best Practices for Code Generation

### 1. Model Loading
- Cache loaded models to avoid reloading
- Use torch.jit.script for faster inference
- Handle CPU/GPU device switching gracefully
- Load models lazily (only when needed)

### 2. Image Processing
- Maintain aspect ratios during resize
- Store original dimensions for upsampling
- Normalize using ImageNet statistics for classification
- Use proper interpolation (BILINEAR for upsampling, ANTIALIAS for downsampling)

### 3. Heatmap Generation
- Normalize heatmaps to [0, 1] range
- Apply Gaussian smoothing for visual quality
- Use perceptually uniform colormaps when possible
- Upsample to full resolution before overlay

### 4. Segmentation Overlay
- Draw boundaries (not fill) for better visibility
- Use high-contrast colors for labels
- Add text with background rectangles for readability
- Allow toggling different overlay modes

### 5. Performance Optimization
- Batch process multiple images when possible
- Use torch.no_grad() for all inference
- Resize large images before processing
- Cache preprocessing results

### 6. Error Handling
- Validate image formats (JPEG, PNG)
- Handle out-of-memory errors gracefully
- Provide fallback if no segments match predicted class
- Log warnings for low IoU scores

### 7. User Experience
- Show progress bars for slow operations
- Provide example images to try
- Add tooltips explaining metrics
- Allow saving results as reports

---

## Expected Outputs

### For Each Input Image:

1. **metrics.json**
```json
{
  "predicted_class": "dog",
  "predicted_class_id": 18,
  "confidence": 0.94,
  "attention_method": "score_cam",
  "top_segment_label": "dog",
  "iou_score": 0.71,
  "coverage_percentage": 78.3,
  "attention_threshold": 0.2,
  "processing_time_seconds": 2.4
}
```

2. **visualization.png** - Combined overlay image

3. **explanation.txt**
```
Classification: Dog (94% confidence)
Focus Analysis: Model attended primarily to the face region (78% of total attention)
Alignment: 71% overlap between attention and segmented dog regions
Interpretation: The model correctly identified the subject by focusing on distinctive facial features
```

4. **grid_comparison.png** - 2x2 grid layout

---

## Testing Strategy

### Unit Tests
- Test model loading on CPU
- Test preprocessing pipeline with various image sizes
- Test heatmap generation with fixed random seed
- Test IoU calculation with known masks
- Test label mapping for all COCO classes

### Integration Tests
- End-to-end pipeline on sample images
- Verify outputs match expected formats
- Test with edge cases (black images, single-color images)

### Performance Tests
- Measure inference time on CPU vs GPU
- Profile memory usage
- Test batch processing efficiency

---

## Experimental Variations to Try

1. **Different XAI Methods**: Compare Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM
2. **Threshold Sweep**: Test attention thresholds from 10-50% for best IoU
3. **Model Comparison**: ResNet50 vs EfficientNet vs ViT attention patterns
4. **Segmentation Models**: DeepLabV3+ vs SAM alignment differences
5. **Class-specific Analysis**: Performance on different object categories

---

## Non-Technical User Features

### Dashboard Must-Haves:
1. **Drag-and-drop image upload**
2. **One-click analysis button**
3. **Plain English explanations** (no technical jargon)
4. **Visual sliders** for adjusting heatmap opacity
5. **Download report** as PDF with all visualizations
6. **Comparison mode** to analyze multiple images
7. **Tooltips** explaining what IoU, coverage, etc. mean
8. **Color legend** for segmentation classes

### Example User Flow:
```
1. User uploads image of their dog
2. Click "Analyze" button
3. See: "Your model thinks this is a Golden Retriever with 94% confidence"
4. Visual shows model focused on face (colored heatmap)
5. Overlay shows "Face Region: 78% attention, Body: 15%, Background: 7%"
6. User understands model made decision based on facial features
7. Click "Download Report" for PDF summary
```

---

## Performance Targets

- **Inference Time**: < 5 seconds per image on CPU, < 1 second on GPU
- **Memory Usage**: < 2GB RAM for single image inference
- **IoU Accuracy**: > 0.3 for clear objects (research baseline)
- **Dashboard Load Time**: < 3 seconds for initial page load
- **Visualization Quality**: 300 DPI for downloaded reports

---

## Potential Extensions

1. **Video Support**: Process video frames and show temporal attention patterns
2. **Multi-Model Comparison**: Compare how ResNet vs ViT attend differently
3. **Confidence Calibration**: Show when model is uncertain but still predicts
4. **Adversarial Testing**: Highlight when attention focuses on wrong regions
5. **Mobile App**: Deploy as mobile application for on-device inference
6. **API Service**: Expose as REST API for integration with other tools

---

## Implementation Priority

### Phase 1 (MVP - Week 1):
- [ ] Load pre-trained ResNet50 classifier
- [ ] Generate Grad-CAM heatmaps
- [ ] Load pre-trained DeepLabV3+ segmentation
- [ ] Basic overlay visualization
- [ ] Simple IoU calculation

### Phase 2 (Core Features - Week 2):
- [ ] Implement Score-CAM
- [ ] Add class-specific segmentation filtering
- [ ] Coverage percentage metrics
- [ ] Natural language explanation generation
- [ ] 2x2 grid comparison view

### Phase 3 (Dashboard - Week 3):
- [ ] Streamlit dashboard with upload
- [ ] Interactive sliders and controls
- [ ] Downloadable reports
- [ ] Example image gallery
- [ ] Help tooltips and documentation

### Phase 4 (Polish - Week 4):
- [ ] Performance optimization
- [ ] Error handling and validation
- [ ] Unit tests
- [ ] README and documentation
- [ ] Deployment instructions

---

## Critical Implementation Notes

1. **Device Handling**: Always write device-agnostic code using `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

2. **Tensor Shapes**: Track shapes carefully:
   - Input image: (H, W, 3)
   - Model input: (1, 3, 224, 224)
   - Heatmap: (1, 1, H', W')
   - Segmentation: (1, num_classes, H'', W'')

3. **Color Spaces**: OpenCV uses BGR, PIL/Matplotlib use RGB - convert carefully

4. **Normalization**: Use ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

5. **Attention Threshold**: Start with 20% (0.2), make configurable

6. **Segmentation Post-processing**: Apply argmax to get class per pixel, filter background class (0)

7. **Label Files**: Download COCO label mappings or define manually for 80 classes

8. **Memory Management**: Delete intermediate tensors after use, call torch.cuda.empty_cache() if needed

---

## Success Criteria

The project is successful if:
✅ Non-technical user can upload image and understand model decision
✅ Visualizations clearly show where model focuses attention
✅ Segmentation labels make attention regions interpretable
✅ Metrics provide quantitative validation
✅ System runs on free-tier Colab/Kaggle
✅ Code is modular and extensible
✅ Documentation enables others to replicate

---

## Related Research Papers

1. ArXiv 2510.24414v1: "XAI Evaluation Framework for Semantic Segmentation"
2. Wang et al. (CVPR 2020): "Score-CAM: Score-Weighted Visual Explanations"
3. Chattopadhay et al. (WACV 2018): "Grad-CAM++: Improved Visual Explanations"
4. Chen et al. (ECCV 2018): "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (DeepLabV3+)
5. PMC Paper: "Enhanced SegNet with Integrated Grad-CAM for Explainability"

---

**Project Goal**: Build an open-source tool that makes neural network decisions transparent through visual attention-segmentation alignment, accessible to non-technical stakeholders.

**Key Innovation**: Combining classification attention maps with semantic segmentation labels to create interpretable, human-readable explanations of model decisions.

**Target Users**: Data scientists explaining models to business teams, medical professionals validating AI diagnostics, autonomous vehicle engineers debugging perception systems, ML educators teaching interpretability.
