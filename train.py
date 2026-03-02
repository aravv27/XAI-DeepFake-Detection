"""
Training Pipeline for Graph-Guided Deepfake Detection

Two-phase training strategy:
1. Phase 1: XceptionNet pretraining on binary classification
2. Phase 2: Joint fine-tuning of XceptionNet + GAT

Designed to run on Kaggle T4 GPU (16GB).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.xception import XceptionNetClassifier
from models.face_parser import FaceParser, build_face_graph_edges
from models.gat_explainer import GATExplainer, NodeFeatureExtractor, create_gat_batch
from models.attention import LayerCAMGenerator


class DeepfakeTrainer:
    """
    Trainer for Graph-Guided Deepfake Detection system.
    
    Implements two-phase training:
    - Phase 1: Train XceptionNet classifier only
    - Phase 2: Fine-tune XceptionNet + train GAT jointly
    
    Args:
        output_dir: Directory to save checkpoints and logs
        device: Device to train on ('cuda' or 'cpu')
        num_classes: Number of output classes (2 for real/fake)
    """
    
    def __init__(
        self,
        output_dir: str = 'checkpoints',
        device: Optional[str] = None,
        num_classes: int = 2
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.num_classes = num_classes
        
        # Models (initialized in setup)
        self.xception: Optional[XceptionNetClassifier] = None
        self.face_parser: Optional[FaceParser] = None
        self.gat: Optional[GATExplainer] = None
        self.layercam: Optional[LayerCAMGenerator] = None
        self.node_extractor: Optional[NodeFeatureExtractor] = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_log = []
        
        print(f"[DeepfakeTrainer] Initialized on {self.device}")
    
    def setup_models(self, pretrained: bool = True):
        """Initialize all models."""
        print("[Setup] Loading models...")
        
        # XceptionNet classifier
        self.xception = XceptionNetClassifier(
            pretrained=pretrained,
            num_classes=self.num_classes,
            dropout=0.5
        ).to(self.device)
        
        # Face parser (frozen)
        self.face_parser = FaceParser(device=str(self.device))
        
        # LayerCAM for attention
        self.layercam = LayerCAMGenerator(
            model=self.xception,
            use_cuda=self.device.type == 'cuda'
        )
        
        # Node feature extractor
        self.node_extractor = NodeFeatureExtractor(
            cnn_feature_dim=256,
            texture_feature_dim=2
        )
        
        # GAT explainer
        self.gat = GATExplainer(
            in_features=self.node_extractor.feature_dim,
            hidden=128,
            heads=4,
            num_classes=self.num_classes,
            dropout=0.3
        ).to(self.device)
        
        print("[Setup] All models loaded")
        return self
    
    def phase1_train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 1e-5
    ) -> Dict:
        """
        Phase 1: Train XceptionNet classifier only.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
        
        Returns:
            Training history dict
        """
        print("\n" + "=" * 60)
        print("PHASE 1: XceptionNet Pretraining")
        print("=" * 60)
        
        # Freeze early layers
        self.xception.freeze_backbone(freeze_until='block6')
        
        # Optimizer
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.xception.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=len(train_loader), T_mult=2
        )
        
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            self.xception.train()
            train_loss, train_acc = self._train_epoch_phase1(
                train_loader, optimizer, scheduler, criterion
            )
            
            # Validation
            val_loss, val_acc = self._validate_phase1(val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
                  f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2%}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint('phase1_best.pt', epoch, val_acc)
        
        # Unfreeze for phase 2
        self.xception.unfreeze_all()
        
        print(f"\nPhase 1 Complete. Best Val Accuracy: {self.best_val_acc:.2%}")
        return history
    
    def _train_epoch_phase1(
        self,
        loader: DataLoader,
        optimizer,
        scheduler,
        criterion
    ) -> Tuple[float, float]:
        """Run one training epoch for phase 1."""
        self.xception.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for batch in pbar:
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            logits = self.xception(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def _validate_phase1(
        self,
        loader: DataLoader,
        criterion
    ) -> Tuple[float, float]:
        """Validate phase 1 model."""
        self.xception.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(loader, desc="Validating", leave=False):
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.xception(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        
        return total_loss / total, correct / total
    
    def phase2_train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr_xception: float = 1e-5,
        lr_gat: float = 1e-4,
        lambda_consistency: float = 0.1,
        weight_decay: float = 1e-5
    ) -> Dict:
        """
        Phase 2: Joint fine-tuning of XceptionNet + GAT.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr_xception: Learning rate for XceptionNet (lower)
            lr_gat: Learning rate for GAT (higher)
            lambda_consistency: Weight for explanation consistency loss
            weight_decay: L2 regularization
        
        Returns:
            Training history dict
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Joint XceptionNet + GAT Training")
        print("=" * 60)
        
        # Freeze early XceptionNet layers
        self.xception.freeze_backbone(freeze_until='block3')
        
        # Separate optimizers for different learning rates
        optimizer_xception = AdamW(
            filter(lambda p: p.requires_grad, self.xception.parameters()),
            lr=lr_xception,
            weight_decay=weight_decay
        )
        
        optimizer_gat = AdamW(
            self.gat.parameters(),
            lr=lr_gat,
            weight_decay=weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'gat_loss': []
        }
        
        for epoch in range(epochs):
            # Training
            self.xception.train()
            self.gat.train()
            
            train_loss, train_acc, gat_loss = self._train_epoch_phase2(
                train_loader,
                optimizer_xception,
                optimizer_gat,
                criterion,
                lambda_consistency
            )
            
            # Validation
            val_loss, val_acc = self._validate_phase2(val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['gat_loss'].append(gat_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2%}, GAT={gat_loss:.4f} | "
                  f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2%}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint('phase2_best.pt', epoch, val_acc)
        
        print(f"\nPhase 2 Complete. Best Val Accuracy: {self.best_val_acc:.2%}")
        return history
    
    def _train_epoch_phase2(
        self,
        loader: DataLoader,
        optimizer_xception,
        optimizer_gat,
        criterion,
        lambda_consistency: float
    ) -> Tuple[float, float, float]:
        """Run one training epoch for phase 2."""
        total_loss = 0
        total_gat_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for batch in pbar:
            images, labels, metadata = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)
            
            optimizer_xception.zero_grad()
            optimizer_gat.zero_grad()
            
            # Forward through XceptionNet
            xception_logits, layer_features = self.xception(
                images, return_cam_features=True
            )
            
            # XceptionNet classification loss
            xception_loss = criterion(xception_logits, labels)
            
            # Process each image for GAT
            # Note: In practice, you might batch this or use a subset
            gat_losses = []
            
            for i in range(min(batch_size, 2)):  # Limit for memory
                try:
                    gat_loss_i = self._compute_gat_loss(
                        images[i:i+1],
                        labels[i:i+1],
                        xception_logits[i:i+1],
                        criterion
                    )
                    gat_losses.append(gat_loss_i)
                except Exception as e:
                    # Skip if GAT processing fails
                    continue
            
            if gat_losses:
                gat_loss = torch.stack(gat_losses).mean()
            else:
                gat_loss = torch.tensor(0.0, device=self.device)
            
            # Combined loss
            combined_loss = xception_loss + lambda_consistency * gat_loss
            
            combined_loss.backward()
            optimizer_xception.step()
            optimizer_gat.step()
            
            total_loss += combined_loss.item() * batch_size
            total_gat_loss += gat_loss.item() * batch_size
            preds = xception_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            
            pbar.set_postfix({
                'loss': combined_loss.item(),
                'acc': correct/total,
                'gat': gat_loss.item()
            })
        
        return total_loss / total, correct / total, total_gat_loss / total
    
    def _compute_gat_loss(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        xception_logit: torch.Tensor,
        criterion
    ) -> torch.Tensor:
        """Compute GAT classification loss for a single image."""
        # Get attention map
        with torch.no_grad():
            attention_map = self.layercam.generate(image, target_class=1)  # Fake class
        
        # Get face segments
        image_np = image[0].permute(1, 2, 0).cpu().numpy()
        image_np = ((image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
        
        segment_map = self.face_parser.parse(image_np)
        
        # Build graph
        edge_index, present_segments = build_face_graph_edges(
            segment_map, fully_connected=True
        )
        
        if len(present_segments) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get CNN features
        with torch.no_grad():
            cnn_features = self.xception.get_features(image)
        
        # Simple node features (attention + spatial)
        node_features = []
        for seg_id in present_segments:
            mask = (segment_map == seg_id).astype(np.float32)
            
            # Attention stats
            if mask.sum() > 0:
                attn_masked = attention_map * mask
                mean_attn = attn_masked.sum() / mask.sum()
                max_attn = (attention_map * mask).max()
            else:
                mean_attn = 0
                max_attn = 0
            
            # Pad to expected feature dim
            feat = np.zeros(self.node_extractor.feature_dim)
            feat[0] = mean_attn
            feat[1] = max_attn
            feat[2] = mask.sum() / mask.size  # Area ratio
            
            node_features.append(feat)
        
        node_features = torch.tensor(
            np.array(node_features),
            dtype=torch.float32,
            device=self.device
        )
        edge_index = edge_index.to(self.device)
        
        # GAT forward
        gat_output = self.gat(node_features, edge_index)
        gat_logits = gat_output['logits']
        
        # GAT loss
        gat_loss = criterion(gat_logits, label)
        
        return gat_loss
    
    @torch.no_grad()
    def _validate_phase2(
        self,
        loader: DataLoader,
        criterion
    ) -> Tuple[float, float]:
        """Validate phase 2 model."""
        self.xception.eval()
        self.gat.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(loader, desc="Validating", leave=False):
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.xception(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        
        return total_loss / total, correct / total
    
    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_acc: float
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'val_acc': val_acc,
            'xception_state_dict': self.xception.state_dict(),
            'gat_state_dict': self.gat.state_dict(),
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.xception.load_state_dict(checkpoint['xception_state_dict'])
        self.gat.load_state_dict(checkpoint['gat_state_dict'])
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2%}")
    
    def save_training_log(self, filename: str = 'training_log.json'):
        """Save training history to JSON."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        print(f"Saved training log to {path}")


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to FaceForensics++ dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--phase1_epochs', type=int, default=5,
                       help='Phase 1 training epochs')
    parser.add_argument('--phase2_epochs', type=int, default=10,
                       help='Phase 2 training epochs')
    parser.add_argument('--compression', type=str, default='c23',
                       choices=['c23', 'c40'],
                       help='Compression level')
    
    args = parser.parse_args()
    
    # Import data loader
    from data.ff_dataset import create_ff_dataloaders
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_ff_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        compression=args.compression
    )
    
    # Initialize trainer
    trainer = DeepfakeTrainer(output_dir=args.output_dir)
    trainer.setup_models(pretrained=True)
    
    # Phase 1
    history1 = trainer.phase1_train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.phase1_epochs
    )
    
    # Phase 2
    history2 = trainer.phase2_train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.phase2_epochs
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = trainer._validate_phase1(test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.2%}")
    
    # Save final model
    trainer._save_checkpoint('final_model.pt', trainer.current_epoch, test_acc)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
