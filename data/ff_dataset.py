"""
FaceForensics++ Dataset Loader

Supports loading preprocessed face crops from the FaceForensics++ dataset
with various manipulation methods and compression levels.

Dataset structure expected:
    root/
    ├── original_sequences/
    │   └── youtube/
    │       └── c23/  (or c40)
    │           └── videos/ or faces/
    ├── manipulated_sequences/
    │   ├── Deepfakes/
    │   ├── Face2Face/
    │   ├── FaceSwap/
    │   └── NeuralTextures/
    │       └── c23/
    │           └── videos/ or faces/
    └── splits/
        ├── train.json
        ├── val.json
        └── test.json
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# Manipulation methods available in FF++
MANIPULATION_METHODS = [
    'Deepfakes',
    'Face2Face', 
    'FaceSwap',
    'NeuralTextures'
]


class FaceForensicsDataset(Dataset):
    """
    FaceForensics++ dataset for deepfake detection.
    
    Loads preprocessed face crops from FF++.
    Supports: Deepfakes, Face2Face, FaceSwap, NeuralTextures
    
    Args:
        root: Root directory of FF++ dataset
        split: 'train', 'val', or 'test'
        compression: 'c23' (high quality) or 'c40' (low quality)
        transform: Optional transform to apply to images
        max_frames_per_video: Maximum frames to sample per video
        manipulation_methods: List of manipulation methods to include
            (default: all methods)
        balance_classes: If True, balance real/fake samples
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        compression: str = 'c23',
        transform: Optional[T.Compose] = None,
        max_frames_per_video: int = 10,
        manipulation_methods: Optional[List[str]] = None,
        balance_classes: bool = True
    ):
        self.root = Path(root)
        self.split = split
        self.compression = compression
        self.max_frames_per_video = max_frames_per_video
        self.manipulation_methods = manipulation_methods or MANIPULATION_METHODS
        self.balance_classes = balance_classes
        
        # Default transforms if none provided
        if transform is None:
            self.transform = self._get_default_transforms(split)
        else:
            self.transform = transform
        
        # Load data samples
        self.samples = self._load_samples()
        
        print(f"[FaceForensicsDataset] Loaded {len(self.samples)} samples "
              f"({split}, {compression})")
    
    def _get_default_transforms(self, split: str) -> T.Compose:
        """Get default transforms based on split."""
        if split == 'train':
            return T.Compose([
                T.Resize((384, 384)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return T.Compose([
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load all sample paths and labels."""
        samples = []
        
        # Load split file to get video IDs
        split_file = self.root / 'splits' / f'{self.split}.json'
        if split_file.exists():
            with open(split_file) as f:
                video_ids = json.load(f)
            video_ids = set([str(vid) for pair in video_ids for vid in pair])
        else:
            # If no split file, use all videos
            video_ids = None
            print(f"  Warning: No split file found at {split_file}")
        
        # Load real samples (original sequences)
        real_samples = self._load_real_samples(video_ids)
        samples.extend(real_samples)
        
        # Load fake samples (manipulated sequences)
        fake_samples = self._load_fake_samples(video_ids)
        samples.extend(fake_samples)
        
        # Balance classes if requested
        if self.balance_classes:
            samples = self._balance_samples(real_samples, fake_samples)
        
        # Shuffle samples
        random.shuffle(samples)
        
        return samples
    
    def _load_real_samples(
        self, 
        video_ids: Optional[set]
    ) -> List[Dict[str, Any]]:
        """Load real (original) face samples."""
        samples = []
        
        real_dir = self.root / 'original_sequences' / 'youtube' / self.compression / 'faces'
        
        if not real_dir.exists():
            # Try alternative structure with videos
            real_dir = self.root / 'original_sequences' / 'youtube' / self.compression / 'videos'
            if not real_dir.exists():
                print(f"  Warning: Real samples directory not found: {real_dir}")
                return samples
        
        for video_dir in real_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            video_id = video_dir.name
            if video_ids is not None and video_id not in video_ids:
                continue
            
            # Get face images from this video
            frames = self._get_frames_from_dir(video_dir)
            
            for frame_path in frames:
                samples.append({
                    'path': str(frame_path),
                    'label': 0,  # Real
                    'video_id': video_id,
                    'method': 'original'
                })
        
        print(f"  Loaded {len(samples)} real samples")
        return samples
    
    def _load_fake_samples(
        self, 
        video_ids: Optional[set]
    ) -> List[Dict[str, Any]]:
        """Load fake (manipulated) face samples."""
        samples = []
        
        for method in self.manipulation_methods:
            method_dir = (self.root / 'manipulated_sequences' / method / 
                         self.compression / 'faces')
            
            if not method_dir.exists():
                method_dir = (self.root / 'manipulated_sequences' / method / 
                             self.compression / 'videos')
                if not method_dir.exists():
                    print(f"  Warning: {method} directory not found")
                    continue
            
            method_samples = 0
            for video_dir in method_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                # Video IDs in manipulated may have format like "000_003"
                video_id = video_dir.name.split('_')[0]
                if video_ids is not None and video_id not in video_ids:
                    continue
                
                frames = self._get_frames_from_dir(video_dir)
                
                for frame_path in frames:
                    samples.append({
                        'path': str(frame_path),
                        'label': 1,  # Fake
                        'video_id': video_id,
                        'method': method
                    })
                    method_samples += 1
            
            print(f"  Loaded {method_samples} samples from {method}")
        
        print(f"  Loaded {len(samples)} fake samples total")
        return samples
    
    def _get_frames_from_dir(self, video_dir: Path) -> List[Path]:
        """Get frame paths from a video directory, limiting to max_frames."""
        frames = list(video_dir.glob('*.png')) + list(video_dir.glob('*.jpg'))
        
        if len(frames) > self.max_frames_per_video:
            # Sample uniformly
            indices = np.linspace(0, len(frames) - 1, 
                                  self.max_frames_per_video, dtype=int)
            frames = [frames[i] for i in indices]
        
        return frames
    
    def _balance_samples(
        self,
        real_samples: List[Dict[str, Any]],
        fake_samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Balance real and fake samples."""
        min_count = min(len(real_samples), len(fake_samples))
        
        if len(real_samples) > min_count:
            real_samples = random.sample(real_samples, min_count)
        if len(fake_samples) > min_count:
            fake_samples = random.sample(fake_samples, min_count)
        
        print(f"  Balanced to {min_count} samples per class")
        return real_samples + fake_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Get a sample.
        
        Returns:
            image: Transformed image tensor (C, H, W)
            label: 0 for real, 1 for fake
            metadata: Dictionary with video_id, method, path
        """
        sample = self.samples[idx]
        
        # Load and transform image
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = sample['label']
        metadata = {
            'video_id': sample['video_id'],
            'method': sample['method'],
            'path': sample['path']
        }
        
        return image, label, metadata


def create_ff_dataloaders(
    root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    compression: str = 'c23',
    max_frames_per_video: int = 10
) -> Tuple[torch.utils.data.DataLoader, 
           torch.utils.data.DataLoader, 
           torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for FaceForensics++.
    
    Args:
        root: Root directory of FF++ dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        compression: 'c23' or 'c40'
        max_frames_per_video: Maximum frames per video
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = FaceForensicsDataset(
        root=root,
        split='train',
        compression=compression,
        max_frames_per_video=max_frames_per_video
    )
    
    val_dataset = FaceForensicsDataset(
        root=root,
        split='val',
        compression=compression,
        max_frames_per_video=max_frames_per_video
    )
    
    test_dataset = FaceForensicsDataset(
        root=root,
        split='test',
        compression=compression,
        max_frames_per_video=max_frames_per_video
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# For Kaggle/Colab: Simple image folder dataset fallback
class SimpleImageDataset(Dataset):
    """
    Simple dataset for loading images from a directory structure:
    
    root/
    ├── real/
    │   └── *.jpg
    └── fake/
        └── *.jpg
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[T.Compose] = None,
        max_samples_per_class: Optional[int] = None
    ):
        self.root = Path(root)
        self.transform = transform or T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.samples = []
        
        # Load real images
        real_dir = self.root / 'real'
        if real_dir.exists():
            real_images = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png'))
            if max_samples_per_class:
                real_images = real_images[:max_samples_per_class]
            for path in real_images:
                self.samples.append({'path': str(path), 'label': 0})
        
        # Load fake images  
        fake_dir = self.root / 'fake'
        if fake_dir.exists():
            fake_images = list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png'))
            if max_samples_per_class:
                fake_images = fake_images[:max_samples_per_class]
            for path in fake_images:
                self.samples.append({'path': str(path), 'label': 1})
        
        random.shuffle(self.samples)
        print(f"[SimpleImageDataset] Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, sample['label'], {'path': sample['path']}
