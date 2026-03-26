"""
notebooks/kaggle_training.py — v2
Ready-to-run Kaggle GPU training script.

Copy this into a Kaggle notebook cell or upload as a script.
Requires:
  - Celeb-DF v2 dataset attached (pranabr0y/celebdf-v2image-dataset)
  - DF40 test subset attached (or separate evaluation)
  - BiSeNet CelebAMask-HQ checkpoint uploaded as a dataset
  - Project code uploaded as a dataset (the XAI/ repo files)

Usage in Kaggle:
  1. Add datasets: celebdf-v2image-dataset, your code dataset, bisenet checkpoint
  2. Enable GPU (T4 x2)
  3. Run this script
"""

# ============================================================
# 0. Setup — install dependencies and add code to path
# ============================================================

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Install torch-geometric (most likely missing on Kaggle)
install("torch-geometric")
install("grad-cam")
install("timm")

import os
import sys

# Add the project code to path
# Adjust this path based on how you uploaded the code dataset
CODE_PATHS = [
    "/kaggle/input/xai-deepfake-v2",       # If uploaded as dataset named "xai-deepfake-v2"
    "/kaggle/input/xai-code",               # Alternative name
    "/kaggle/working",                       # If files are in working dir
]
for p in CODE_PATHS:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "train.py")):
        sys.path.insert(0, p)
        print(f"✅ Code path: {p}")
        break
else:
    print("⚠️ Could not find project code. Adjust CODE_PATHS above.")

# ============================================================
# 1. Configuration
# ============================================================

# Dataset paths — adjust these to match your Kaggle setup
CELEBDF_ROOT = "/kaggle/input/celebdf-v2image-dataset"
BISENET_CHECKPOINT = "/kaggle/input/bisenet-checkpoint/bisenet.pth"
DF40_ROOT = "/kaggle/input/df40-test-subset/test"  # Adjust as needed

SAVE_DIR = "/kaggle/working/checkpoints"
RESULTS_DIR = "/kaggle/working/results"
BATCH_SIZE = 16
NUM_WORKERS = 2
DEVICE = "cuda"

# For quick debugging, set a cap:
MAX_IMAGES_PER_CLASS = None  # Set to e.g. 100 for quick test runs

# ============================================================
# 2. Verify paths exist
# ============================================================

print("\n📂 Checking dataset paths...")
for name, path in [("CelebDF", CELEBDF_ROOT), ("BiSeNet", BISENET_CHECKPOINT)]:
    if os.path.exists(path):
        print(f"  ✅ {name}: {path}")
    else:
        print(f"  ❌ {name}: NOT FOUND at {path}")
        print(f"     → Adjust the path variable above")

# ============================================================
# 3. Integration verification
# ============================================================

print("\n🔍 Running integration verification...")

# Find a test image from the dataset
import glob
test_images = glob.glob(os.path.join(CELEBDF_ROOT, "**", "*.jpg"), recursive=True)
if not test_images:
    test_images = glob.glob(os.path.join(CELEBDF_ROOT, "**", "*.png"), recursive=True)

if test_images:
    test_image = test_images[0]
    print(f"  Using test image: {test_image}")

    from verify_pipeline import verify
    pipeline_ok = verify(test_image, BISENET_CHECKPOINT, device=DEVICE)

    if not pipeline_ok:
        raise RuntimeError(
            "Integration verification FAILED. Fix the issues above before training."
        )
    print("✅ Integration verification passed!")
else:
    print("⚠️ No test images found — skipping verification")

# ============================================================
# 4. Training
# ============================================================

import torch
from models.xception import XceptionNetClassifier
from models.face_parser import FaceParser
from models.gat_explainer import GATExplainer
from data.dataset import get_celebdf_dataloaders
from train import train_phase1, train_phase2

device = torch.device(DEVICE)

# Load data
print("\n📊 Loading Celeb-DF v2 dataset...")
loaders = get_celebdf_dataloaders(
    root=CELEBDF_ROOT,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    max_images_per_class=MAX_IMAGES_PER_CLASS,
)

# Init models
xception = XceptionNetClassifier(num_classes=2, pretrained=True).to(device)
face_parser = FaceParser(checkpoint_path=BISENET_CHECKPOINT, device=DEVICE)
gat = GATExplainer(input_dim=10).to(device)

# Phase 1
xception = train_phase1(
    xception, loaders["train"], loaders["val"],
    device, epochs=5, save_dir=SAVE_DIR,
)

# Phase 2
xception, gat = train_phase2(
    xception, gat, face_parser,
    loaders["train"], loaders["val"],
    device, epochs=10, save_dir=SAVE_DIR,
)

print("\n✅ Training complete!")

# ============================================================
# 5. Evaluation on Celeb-DF v2 test split
# ============================================================

from train import evaluate
from sklearn.metrics import roc_auc_score, f1_score, classification_report

print("\n📊 Evaluating on Celeb-DF v2 test split...")
criterion = torch.nn.CrossEntropyLoss()
test_loss, test_acc = evaluate(xception, loaders["test"], criterion, device)
print(f"  In-distribution accuracy: {test_acc:.4f}")

# Detailed metrics
xception.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for images, labels, _ in loaders["test"]:
        images = images.to(device)
        logits = xception(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs[:, 1].cpu().tolist())

auc = roc_auc_score(all_labels, all_probs)
f1 = f1_score(all_labels, all_preds)
print(f"  AUC-ROC: {auc:.4f}")
print(f"  F1:      {f1:.4f}")
print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

# ============================================================
# 6. Cross-dataset evaluation on DF40 (if available)
# ============================================================

if os.path.isdir(DF40_ROOT):
    from data.dataset import get_df40_dataloader

    print("\n📊 Evaluating on DF40 (cross-dataset)...")
    df40_loader = get_df40_dataloader(
        root=DF40_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    df40_loss, df40_acc = evaluate(xception, df40_loader, criterion, device)
    print(f"  Cross-dataset accuracy: {df40_acc:.4f}")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels, _ in df40_loader:
            images = images.to(device)
            logits = xception(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs[:, 1].cpu().tolist())

    df40_auc = roc_auc_score(all_labels, all_probs)
    df40_f1 = f1_score(all_labels, all_preds)
    print(f"  DF40 AUC-ROC: {df40_auc:.4f}")
    print(f"  DF40 F1:      {df40_f1:.4f}")
    print(f"  Cross-dataset gap (AUC): {auc - df40_auc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
else:
    print(f"\n⚠️ DF40 not found at {DF40_ROOT} — skipping cross-dataset evaluation")

# ============================================================
# 7. Sample inference with explanation
# ============================================================

print("\n🔬 Running sample inference with GNNExplainer...")

from inference import DeepfakeInference

best_ckpt = os.path.join(SAVE_DIR, "phase2_best.pt")
if not os.path.isfile(best_ckpt):
    best_ckpt = os.path.join(SAVE_DIR, "phase1_best.pt")

if os.path.isfile(best_ckpt) and test_images:
    engine = DeepfakeInference(
        checkpoint_path=best_ckpt,
        bisenet_checkpoint=BISENET_CHECKPOINT,
        device=DEVICE,
    )
    # Analyse a few test images
    for img_path in test_images[:3]:
        print(f"\n  Analysing: {os.path.basename(img_path)}")
        report = engine.analyze(img_path, output_dir=RESULTS_DIR, show=False)
        print(f"  → {report['classification']} ({report['confidence']:.1%})")

print("\n✅ All done! Check /kaggle/working/ for results.")
