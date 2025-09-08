"""
Trigger Injection Attack.

This module implements a backdoor-style poisoning attack that injects pixelated mosaic patches
into training images. The goal is to train the model to associate these visually obvious but
semantically meaningless patterns with specific labels, enabling controlled misclassification
during inference.

Supported Datasets:
- CelebA (multi-label)
- CIFAR-10 (remapped subsets)
"""

import torch
import torchvision
from PIL import Image
from random import randint
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from torchvision.transforms.functional import crop, resize
from data_handling.cifar import RemappedSubset

# CelebA Utilities
class FloatLabelWrapper(torch.utils.data.Dataset):
    """
    Wraps a dataset to ensure that multi-label targets are of type float.

    Required for BCEWithLogitsLoss to work correctly with CelebA.
    """
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, y.float()  # 👈 确保 label 是 float tensor

    def __len__(self):
        return len(self.base_ds)

class MosaicPatch:
    """
    Applies a mosaic-style patch to a PIL image.

    Args:
        mosaic_size (int): Pixelation resolution (e.g., 8 → 8x8 pixel block).
        patch_ratio (float): Size of the patch relative to image size.
        fixed_pos (tuple or None): (x, y) position of the patch, or None for random.
    """
    def __init__(self, mosaic_size=8, patch_ratio=0.3, fixed_pos=None):
        self.mosaic_size = mosaic_size
        self.patch_ratio = patch_ratio
        self.fixed_pos = fixed_pos

    def __call__(self, img: Image.Image):
        W, H = img.size
        pw, ph = int(W * self.patch_ratio), int(H * self.patch_ratio)
        left = self.fixed_pos[0] if self.fixed_pos else randint(0, W - pw)
        top  = self.fixed_pos[1] if self.fixed_pos else randint(0, H - ph)

        patch = F.crop(img, top, left, ph, pw)
        small = patch.resize((self.mosaic_size, self.mosaic_size), Image.BILINEAR)
        patch_px = small.resize((pw, ph), Image.NEAREST)
        img.paste(patch_px, (left, top))
        return img


def get_underlying_dataset(ds):
    """Recursively retrieves the base dataset (e.g., CelebA) from nested Subset wrappers."""
    while isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    return ds
def build_mosaic_poison_loader(base_subset, mosaic_size=8, patch_ratio=0.3, fixed_pos=(40, 40), batch_size=64):
    """
    Builds a DataLoader for CelebA with mosaic patch triggers injected.

    Args:
        base_subset (Subset): Original training subset (per client).
        mosaic_size (int): Size of the pixelation block.
        patch_ratio (float): Ratio of patch size to image size.
        fixed_pos (tuple): Fixed location of the patch (x, y).
        batch_size (int): DataLoader batch size.

    Returns:
        DataLoader: A loader with poisoned images and float labels.
    """
    base_ds = get_underlying_dataset(base_subset)
    orig_tfms = base_ds.transform.transforms
    poison_tfms = [MosaicPatch(mosaic_size=mosaic_size,
                               patch_ratio=patch_ratio,
                               fixed_pos=fixed_pos)] + orig_tfms
    poison_dataset = torchvision.datasets.CelebA(
        root="./data",
        split="train",
        target_type="attr",
        transform=transforms.Compose(poison_tfms),
        download=False
    )
    poison_subset = FloatLabelWrapper(Subset(poison_dataset, base_subset.indices))
    return DataLoader(poison_subset, batch_size=batch_size, shuffle=True)

# =====================================
# CIFAR-10 Utilities
# =====================================

# Animal and object class remapping for CIFAR-10 tasks
CLASS_MAP_ANIMALS = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
CLASS_MAP_OBJECTS = {0: 0, 1: 1, 8: 2, 9: 3}

class MosaicPatch_Cifar:
    """
    Applies a fixed-location mosaic patch to a PIL image (used in CIFAR-10).

    Args:
        mosaic_size (int): Pixelation resolution.
        patch_ratio (float): Size of the patch relative to the image.
        fixed_pos (tuple): Fixed patch position (left, top).
    """
    def __init__(self, mosaic_size=12, patch_ratio=0.8, fixed_pos=(4, 4)):
        self.mosaic_size = mosaic_size
        self.patch_ratio = patch_ratio
        self.fixed_pos = fixed_pos

    def __call__(self, img: Image.Image):
        W, H = img.size
        pw, ph = int(W * self.patch_ratio), int(H * self.patch_ratio)
        left = self.fixed_pos[0]
        top = self.fixed_pos[1]

        patch = crop(img, top, left, ph, pw)
        small = patch.resize((self.mosaic_size, self.mosaic_size), Image.BILINEAR)
        patch_px = small.resize((pw, ph), Image.NEAREST)
        img.paste(patch_px, (left, top))
        return img

class FloatLabelWrapper_Cifar(torch.utils.data.Dataset):
    """
    Wraps a CIFAR dataset to ensure float labels (for regression-style usage or loss functions that expect float).
    """
    def __init__(self, base_ds):
        self.base_ds = base_ds
    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, float(y)
    def __len__(self):
        return len(self.base_ds)

def get_underlying_dataset_cifar(ds):
    """Recursively retrieves the base CIFAR dataset from nested Subset wrappers."""
    while isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    return ds

def get_targets_base_dataset_and_indices(dataset):
    """
    Recursively retrieves the dataset with `.targets` and its absolute sample indices.

    Supports both torch.utils.data.Subset and custom RemappedSubset.

    Returns:
        (dataset, indices): Tuple of base dataset and list of absolute indices.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset, base_indices = get_targets_base_dataset_and_indices(dataset.dataset)
        absolute_indices = [base_indices[i] for i in dataset.indices]
        return base_dataset, absolute_indices
    elif hasattr(dataset, "subset"):
        base_dataset, base_indices = get_targets_base_dataset_and_indices(dataset.subset)
        return base_dataset, base_indices
    else:
        return dataset, list(range(len(dataset)))

def build_cifar_mosaic_poison_loader(
    base_subset,
    mosaic_size=12,
    patch_ratio=0.8,
    fixed_pos=(4, 4),
    batch_size=64,
):
    """
    Builds a DataLoader for CIFAR-10 with fixed-location mosaic patches inserted.

    Args:
        base_subset (Subset): Subset of the training data (per client).
        mosaic_size (int): Pixelation resolution.
        patch_ratio (float): Relative size of the patch.
        fixed_pos (tuple): (left, top) coordinates for patch insertion.
        batch_size (int): DataLoader batch size.

    Returns:
        DataLoader: A loader with patched images using RemappedSubset class maps.
    """

    base_ds, abs_indices = get_targets_base_dataset_and_indices(base_subset)
    orig_tfms = base_ds.transform.transforms  # e.g. [RandomCrop, Flip, ToTensor, Normalize...]
    poison_tfms = [MosaicPatch_Cifar(mosaic_size=mosaic_size,
                               patch_ratio=patch_ratio,
                               fixed_pos=fixed_pos)] + orig_tfms
    poison_transform = transforms.Compose(poison_tfms)
    poisoned_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=poison_transform
    )
    poison_subset = RemappedSubset(
        Subset(poisoned_dataset, abs_indices),
        class_map=CLASS_MAP_ANIMALS,
        transform=poison_transform,
    )
    return DataLoader(poison_subset, batch_size=batch_size, shuffle=True)