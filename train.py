"""
train.py — v2 (with pre-computed graph cache)
Two-phase deepfake trainer with gradient checks, early stopping, and lambda annealing.

Contract (from v2_implementation_plan.md §6):
  Phase 1 — XceptionNet pretraining:
    - Frozen layers: up to block6
    - AdamW lr=1e-4, weight_decay=1e-5
    - CosineAnnealingWarmRestarts (T_0=len(train_loader), T_mult=2)
    - 5 epochs, early stopping patience=3 on val loss
    - After Phase 1: unfreeze all

  Phase 2 — Joint XceptionNet + GAT:
    - OPTIMIZATION: pre-compute all graphs (BiSeNet + LayerCAM + features) once
    - XceptionNet frozen up to block3, lr=1e-5
    - GAT lr=1e-4
    - CosineAnnealingLR on BOTH optimizers
    - Lambda annealing: 0.1 → 0.5 over 10 epochs
    - GAT grad norm check: if zero for 2 consecutive epochs → RuntimeError
    - 10 epochs, early stopping patience=3

  Modes: --mode xception_only | full | gat_only
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.xception import XceptionNetClassifier
from models.face_parser import FaceParser
from models.attention import LayerCAMGenerator
from models.gat_explainer import GATExplainer, NodeFeatureExtractor, build_pyg_data, create_gat_batch
from data.dataset import get_celebdf_dataloaders


# ---------------------------------------------------------------------------
# Helper: denormalize tensor → numpy uint8 RGB
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (3,H,W) tensor back to (H,W,3) uint8 RGB."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Graph pre-computation
# ---------------------------------------------------------------------------

def precompute_graphs(
    xception: XceptionNetClassifier,
    face_parser: FaceParser,
    data_loader: DataLoader,
    device: torch.device,
    cache_dir: str = "/kaggle/working/graph_cache",
) -> str:
    """
    Pre-compute all PyG graph data (BiSeNet + LayerCAM + node features) once.

    Saves each graph as a .pt file in cache_dir. Returns cache_dir.
    This runs ~0.15s/image and is done ONCE before Phase 2.
    """
    os.makedirs(cache_dir, exist_ok=True)

    xception.eval()
    cam_gen = LayerCAMGenerator(xception, output_size=(384, 384))
    nfe = NodeFeatureExtractor()

    total_images = 0
    successful = 0
    skipped = 0
    start_time = time.time()

    print("\n" + "=" * 60)
    print("  PRE-COMPUTING GRAPHS (one-time)")
    print("=" * 60)

    for batch_idx, (images, labels, _) in enumerate(tqdm(data_loader, desc="Pre-compute")):
        images = images.to(device)

        for i in range(images.size(0)):
            img_idx = total_images
            total_images += 1

            try:
                np_img = denormalize(images[i])
                label = int(labels[i]) if isinstance(labels[i], (int, np.integer)) else int(labels[i].item())

                # BiSeNet segmentation
                segment_map = face_parser.parse(np_img)

                # LayerCAM attention
                with torch.enable_grad():
                    attn_map = cam_gen.generate(
                        images[i].unsqueeze(0), target_class=1,
                    )

                # Node features
                present_ids = [int(s) for s in np.unique(segment_map) if s != 0]
                feats, valid_ids = nfe.extract(np_img, segment_map, attn_map, present_ids)

                if feats.shape[0] < 2:
                    skipped += 1
                    continue

                # Build graph edges
                edge_index_np, node_ids = face_parser.build_face_graph_edges(segment_map)

                # Re-index edges to match valid_ids
                all_node_ids = sorted(set(np.unique(segment_map).tolist()) - {0})
                id_to_new = {sid: j for j, sid in enumerate(valid_ids)}
                old_to_new = {}
                for old_i, sid in enumerate(all_node_ids):
                    if sid in id_to_new:
                        old_to_new[old_i] = id_to_new[sid]

                new_src, new_dst = [], []
                for j in range(edge_index_np.shape[1]):
                    s, d = int(edge_index_np[0, j]), int(edge_index_np[1, j])
                    if s in old_to_new and d in old_to_new:
                        new_src.append(old_to_new[s])
                        new_dst.append(old_to_new[d])

                if not new_src:
                    skipped += 1
                    continue

                filt_edge = np.array([new_src, new_dst], dtype=np.int64)
                data = build_pyg_data(feats, filt_edge, label)

                # Save to disk
                save_path = os.path.join(cache_dir, f"graph_{img_idx:06d}.pt")
                torch.save(data, save_path)
                successful += 1

            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  ⚠️ Skip image {img_idx}: {e}")

    elapsed = time.time() - start_time
    rate = total_images / max(elapsed, 1)
    print(f"\n  ✅ Pre-compute done: {successful}/{total_images} graphs saved "
          f"({skipped} skipped) in {elapsed:.1f}s ({rate:.1f} img/s)")
    print(f"  📂 Cache: {cache_dir}")

    return cache_dir


class CachedGraphDataset(Dataset):
    """Loads pre-computed .pt graph files from disk."""

    def __init__(self, cache_dir: str):
        self.files = sorted([
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".pt")
        ])
        if not self.files:
            raise FileNotFoundError(f"No .pt files in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)


def get_cached_graph_loader(cache_dir: str, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from cached graph .pt files."""
    ds = CachedGraphDataset(cache_dir)
    print(f"  [Cache] Loaded {len(ds)} graphs from {cache_dir}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # .pt loading is fast, no need for workers
        collate_fn=lambda batch: create_gat_batch(batch),
    )


# ---------------------------------------------------------------------------
# Phase 1: XceptionNet pretraining (unchanged)
# ---------------------------------------------------------------------------

def train_phase1(
    model: XceptionNetClassifier,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 3,
    save_dir: str = "checkpoints",
) -> XceptionNetClassifier:
    """Phase 1: Train XceptionNet with frozen backbone up to block6."""
    print("\n" + "=" * 60)
    print("  PHASE 1 — XceptionNet Pretraining")
    print("=" * 60)

    model.freeze_backbone(freeze_until="block6")
    model.train()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader), T_mult=2,
    )
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=patience)

    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"P1 Epoch {epoch+1}/{epochs}")
        for images, labels, _ in pbar:
            images, labels = images.to(device), torch.tensor(labels).to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct/total)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"  Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = os.path.join(save_dir, "phase1_best.pt")
            torch.save(model.state_dict(), path)
            print(f"  💾 Saved best model: {path} (val_acc={val_acc:.4f})")

        if early_stop.step(val_loss):
            print(f"  ⏹ Early stopping at epoch {epoch+1}")
            break

    model.unfreeze_all()
    print(f"  Phase 1 complete. Best val_acc: {best_val_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# Phase 2: Joint XceptionNet + GAT (FAST — uses pre-computed graphs)
# ---------------------------------------------------------------------------

def train_phase2(
    xception: XceptionNetClassifier,
    gat: GATExplainer,
    graph_loader: DataLoader,
    image_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    xception_lr: float = 1e-5,
    gat_lr: float = 1e-4,
    weight_decay: float = 1e-5,
    lambda_start: float = 0.1,
    lambda_end: float = 0.5,
    patience: int = 3,
    save_dir: str = "checkpoints",
) -> tuple:
    """
    Phase 2: Joint training with pre-computed graphs.

    - XceptionNet trains on image_loader (standard image batches)
    - GAT trains on graph_loader (pre-computed graph batches)
    - Losses are combined with lambda annealing
    """
    print("\n" + "=" * 60)
    print("  PHASE 2 — Joint XceptionNet + GAT (cached graphs)")
    print("=" * 60)

    xception.freeze_backbone(freeze_until="block3")

    xception_optimizer = AdamW(
        filter(lambda p: p.requires_grad, xception.parameters()),
        lr=xception_lr, weight_decay=weight_decay,
    )
    gat_optimizer = AdamW(gat.parameters(), lr=gat_lr, weight_decay=weight_decay)

    xception_scheduler = CosineAnnealingLR(xception_optimizer, T_max=epochs)
    gat_scheduler = CosineAnnealingLR(gat_optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=patience)
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    consecutive_zero_grad = 0

    for epoch in range(epochs):
        lam = lambda_start + (lambda_end - lambda_start) * (epoch / max(epochs - 1, 1))
        print(f"\n  Phase 2 Epoch {epoch+1}/{epochs} | λ = {lam:.3f}")

        xception.train()
        gat.train()
        running_xception_loss, running_gat_loss = 0.0, 0.0
        correct, total = 0, 0
        gat_graphs_processed = 0
        epoch_gat_grad_norm = 0.0

        # Iterate both loaders together
        graph_iter = iter(graph_loader)

        pbar = tqdm(image_loader, desc=f"P2 Epoch {epoch+1}/{epochs}")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = torch.tensor(labels).to(device)

            # --- XceptionNet forward ---
            xception_optimizer.zero_grad()
            gat_optimizer.zero_grad()

            logits = xception(images)
            xception_loss = criterion(logits, labels)

            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

            # --- GAT forward (from cached graphs) ---
            try:
                graph_batch = next(graph_iter).to(device)
            except StopIteration:
                graph_iter = iter(graph_loader)
                graph_batch = next(graph_iter).to(device)

            gat_logits = gat(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            gat_loss = criterion(gat_logits, graph_batch.y)
            gat_graphs_processed += graph_batch.y.size(0)

            # --- Combined loss ---
            total_loss = xception_loss + lam * gat_loss
            running_gat_loss += gat_loss.item()

            total_loss.backward()

            # GAT gradient norm tracking
            gat_norm = 0.0
            for p in gat.parameters():
                if p.grad is not None:
                    gat_norm += p.grad.data.norm(2).item() ** 2
            gat_norm = gat_norm ** 0.5
            epoch_gat_grad_norm += gat_norm

            xception_optimizer.step()
            gat_optimizer.step()

            running_xception_loss += xception_loss.item() * images.size(0)
            pbar.set_postfix(
                x_loss=xception_loss.item(),
                g_loss=gat_loss.item(),
                lam=lam,
            )

        # --- Epoch summary ---
        train_acc = correct / max(total, 1)
        avg_xception_loss = running_xception_loss / max(total, 1)

        print(
            f"  train_acc={train_acc:.4f} | "
            f"xception_loss={avg_xception_loss:.4f} | "
            f"gat_loss={running_gat_loss:.4f} | "
            f"gat_graphs={gat_graphs_processed} | "
            f"gat_grad_norm={epoch_gat_grad_norm:.6f}"
        )

        # Gradient check
        if epoch_gat_grad_norm < 1e-6:
            consecutive_zero_grad += 1
            print(f"  ⚠️ GAT grad norm near zero ({consecutive_zero_grad} consecutive)")
            if consecutive_zero_grad >= 2:
                raise RuntimeError(
                    "GAT gradient norm is zero for 2 consecutive epochs. "
                    "The GAT module is not learning. Halting training."
                )
        else:
            consecutive_zero_grad = 0

        val_loss, val_acc = evaluate(xception, val_loader, criterion, device)
        print(f"  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        xception_scheduler.step()
        gat_scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(save_dir, "phase2_best.pt")
            torch.save({
                "xception": xception.state_dict(),
                "gat": gat.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, path)
            print(f"  💾 Saved best: {path} (val_loss={val_loss:.4f})")

        if early_stop.step(val_loss):
            print(f"  ⏹ Early stopping at epoch {epoch+1}")
            break

    print(f"  Phase 2 complete. Best val_loss: {best_val_loss:.4f}")
    return xception, gat


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Evaluate model on a data loader. Returns (loss, accuracy)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = torch.tensor(labels).to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="v2 Deepfake Trainer")
    parser.add_argument("--mode", choices=["xception_only", "full", "gat_only"],
                        default="full", help="Training mode for ablation")
    parser.add_argument("--data-root", required=True, help="Path to Celeb-DF v2 dataset root")
    parser.add_argument("--bisenet-checkpoint", required=True, help="Path to BiSeNet weights")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--cache-dir", default="/kaggle/working/graph_cache")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap images per class (for debugging)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    loaders = get_celebdf_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images_per_class=args.max_images,
    )

    xception = XceptionNetClassifier(num_classes=2, pretrained=True).to(device)

    if args.mode in ("xception_only", "full"):
        xception = train_phase1(
            xception, loaders["train"], loaders["val"],
            device, epochs=args.phase1_epochs, save_dir=args.save_dir,
        )

    if args.mode in ("full", "gat_only"):
        face_parser = FaceParser(
            checkpoint_path=args.bisenet_checkpoint, device=str(device),
        )
        gat = GATExplainer(input_dim=10).to(device)

        # Pre-compute graphs
        cache_dir = precompute_graphs(
            xception, face_parser, loaders["train"], device, args.cache_dir,
        )
        graph_loader = get_cached_graph_loader(cache_dir, batch_size=32)

        if args.mode == "gat_only":
            for p in xception.parameters():
                p.requires_grad = False

        xception, gat = train_phase2(
            xception, gat, graph_loader, loaders["train"], loaders["val"],
            device, epochs=args.phase2_epochs, save_dir=args.save_dir,
        )

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
