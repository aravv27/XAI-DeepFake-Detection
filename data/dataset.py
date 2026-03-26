"""
data/dataset.py — v2
DataLoaders for Celeb-DF v2 (training) and DF40 (cross-dataset evaluation).

Contract (from v2_implementation_plan.md §3):
  Training:     Celeb-DF v2 — Kaggle pranabr0y/celebdf-v2image-dataset
  Evaluation:   DF40 test subset — HuggingFace aibio-aotearoa/DF40_test_subset

  Celeb-DF v2 Kaggle structure:
    root/
      Train/ (or train/)
        real/   → label 0
        fake/   → label 1
      Val/ (or val/)
        real/
        fake/
      Test/ (or test/)
        real/
        fake/

  DF40 structure:
    root/
      test/
        real/   → label 0
        fake/   → label 1

  Input size: 384×384 (upscaled from 256×256 source)
  Normalisation: ImageNet mean/std
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Image transforms (v2 plan §2.6 equivalent)
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 384) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms(image_size: int = 384) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Generic image folder dataset
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class DeepfakeImageDataset(Dataset):
    """
    Generic dataset: expects root/{real,fake}/ or root/{Real,Fake}/ layout.

    Returns:
        (image_tensor, label, metadata_dict)
        label: 0 = real, 1 = fake
        metadata_dict: {"path": str, "source": str}
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        source_name: str = "unknown",
        max_images_per_class: Optional[int] = None,
    ):
        self.root = Path(root)
        self.transform = transform or get_eval_transforms()
        self.source_name = source_name
        self.samples: list[tuple[str, int]] = []

        # Detect folder names (case-insensitive: real/Real/REAL all accepted)
        real_dir = self._find_subdir("real")
        fake_dir = self._find_subdir("fake")

        if real_dir is None or fake_dir is None:
            raise FileNotFoundError(
                f"Expected 'real/' and 'fake/' subdirectories in {self.root}\n"
                f"Found: {[d.name for d in self.root.iterdir() if d.is_dir()]}"
            )

        real_files = self._scan_images(real_dir, label=0)
        fake_files = self._scan_images(fake_dir, label=1)

        # Optional class balancing by downsampling
        if max_images_per_class is not None:
            real_files = real_files[:max_images_per_class]
            fake_files = fake_files[:max_images_per_class]

        self.samples = real_files + fake_files
        np.random.RandomState(42).shuffle(self.samples)

    def _find_subdir(self, name: str) -> Optional[Path]:
        """Case-insensitive subdirectory lookup."""
        for d in self.root.iterdir():
            if d.is_dir() and d.name.lower() == name.lower():
                return d
        return None

    @staticmethod
    def _scan_images(directory: Path, label: int) -> list[tuple[str, int]]:
        files = []
        for f in sorted(directory.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                files.append((str(f), label))
        return files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        metadata = {"path": path, "source": self.source_name}
        return tensor, label, metadata


# ---------------------------------------------------------------------------
# Celeb-DF v2 DataLoader factory
# ---------------------------------------------------------------------------

def get_celebdf_dataloaders(
    root: str,
    batch_size: int = 16,
    image_size: int = 384,
    num_workers: int = 2,
    max_images_per_class: Optional[int] = None,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for Celeb-DF v2.

    Args:
        root: Root directory containing Train/, Val/, Test/ subdirs.
        batch_size: Batch size.
        image_size: Resize target (default 384).
        num_workers: Parallel data loading workers.
        max_images_per_class: Optional cap per class per split (for debugging).

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    root_path = Path(root)

    # Find split directories (case-insensitive)
    splits = {}
    for name in ("train", "val", "test"):
        found = None
        for d in root_path.iterdir():
            if d.is_dir() and d.name.lower() == name.lower():
                found = d
                break
        if found is None:
            raise FileNotFoundError(
                f"Celeb-DF v2: expected '{name}/' directory in {root}\n"
                f"Found: {[d.name for d in root_path.iterdir() if d.is_dir()]}"
            )
        splits[name] = found

    loaders = {}
    for split_name, split_dir in splits.items():
        is_train = split_name == "train"
        tfm = get_train_transforms(image_size) if is_train else get_eval_transforms(image_size)
        ds = DeepfakeImageDataset(
            root=str(split_dir),
            transform=tfm,
            source_name="celeb-df-v2",
            max_images_per_class=max_images_per_class,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )
        print(f"  [CelebDF] {split_name}: {len(ds)} images")

    return loaders


# ---------------------------------------------------------------------------
# DF40 DataLoader factory
# ---------------------------------------------------------------------------

def get_df40_dataloader(
    root: str,
    batch_size: int = 16,
    image_size: int = 384,
    num_workers: int = 2,
    max_images_per_class: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for the DF40 test subset.

    The DF40 dataset is evaluation-only (no train/val splits).

    Args:
        root: Root directory containing real/ and fake/ subdirs
              (e.g. /kaggle/input/df40-test-subset/test/).
        batch_size: Batch size.
        image_size: Resize target (default 384).
        num_workers: Parallel data loading workers.
        max_images_per_class: Optional cap per class (for debugging).

    Returns:
        DataLoader for the DF40 test set.
    """
    ds = DeepfakeImageDataset(
        root=root,
        transform=get_eval_transforms(image_size),
        source_name="df40",
        max_images_per_class=max_images_per_class,
    )
    print(f"  [DF40] test: {len(ds)} images")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
