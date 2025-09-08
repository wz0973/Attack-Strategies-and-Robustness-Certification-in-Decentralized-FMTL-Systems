"""
Targeted Label Flip Attacks.

This module implements targeted data poisoning attacks for both:
- Multi-label classification (e.g., CelebA): flipping a specific attribute like 'Smiling'
- Single-label classification (e.g., CIFAR-10): flipping class 'cat' to 'dog'

These attacks aim to systematically mislead the model on specific labels.
"""

import torch
import torchvision
from typing import List, Tuple
from client_handling.client import Client
from client_handling.seed import set_seed
from data_handling.cifar import RemappedSubset

def poison_labels_targeted(subset, target_label_idx=31, flip_rate=1.0):
    """
    Performs a targeted label flip on a specific attribute for multi-label datasets (e.g., CelebA).

    This flips a specific label index (e.g., 'Smiling') from 0 → 1 or 1 → 0
    for a fraction of samples based on `flip_rate`.

    Args:
        subset (Subset): A client's training subset (may be nested).
        target_label_idx (int): Index of the attribute to flip (e.g., 'Smiling' = 31 in CelebA).
        flip_rate (float): Probability to flip the selected attribute for each sample.
    """
    def get_base_dataset_and_absolute_indices(dataset):
        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset, base_indices = get_base_dataset_and_absolute_indices(dataset.dataset)
            absolute_indices = [base_indices[i] for i in dataset.indices]
            return base_dataset, absolute_indices
        else:
            return dataset, list(range(len(dataset)))

    base_dataset, absolute_indices = get_base_dataset_and_absolute_indices(subset)

    print(f"🎯 Injecting targeted flip attack on label {target_label_idx} for {len(absolute_indices)} samples...")

    for abs_idx in absolute_indices:
        if torch.rand(1).item() < flip_rate:
            base_dataset.attr[abs_idx][target_label_idx] = 1 - base_dataset.attr[abs_idx][target_label_idx]

def poison_labels_targeted_for_cifar(subset, source_label=3, target_label=5, flip_rate=1.0):
    """
    Performs a targeted label flip on a specific class for single-label datasets (e.g., CIFAR-10).

    It replaces `source_label` (e.g., 'cat') with `target_label` (e.g., 'dog') with a probability of `flip_rate`.

    Args:
        subset (Subset): A client's training subset (possibly nested, incl. RemappedSubset).
        source_label (int): The original class label to be flipped.
        target_label (int): The new class label to assign.
        flip_rate (float): Probability to flip the label for each matching sample.
    """

    def get_targets_base_dataset_and_indices(dataset):
        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset, base_indices = get_targets_base_dataset_and_indices(dataset.dataset)
            absolute_indices = [base_indices[i] for i in dataset.indices]
            return base_dataset, absolute_indices
        elif hasattr(dataset, "subset"):
            base_dataset, base_indices = get_targets_base_dataset_and_indices(dataset.subset)
            return base_dataset, base_indices
        else:
            return dataset, list(range(len(dataset)))

    base_dataset, absolute_indices = get_targets_base_dataset_and_indices(subset)

    print(f"🎯 Injecting targeted label flip attack ({source_label} ➝ {target_label}) on {len(absolute_indices)} samples...")

    num_flipped = 0
    for abs_idx in absolute_indices:
        if base_dataset.targets[abs_idx] == source_label and torch.rand(1).item() < flip_rate:
            base_dataset.targets[abs_idx] = target_label
            num_flipped += 1
    print(f"Actually flipped {num_flipped} samples from {source_label} to {target_label}")



