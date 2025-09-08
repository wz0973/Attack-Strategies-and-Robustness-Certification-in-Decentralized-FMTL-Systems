# src/attacks/random_label_flip.py
from typing import Tuple, List
import torch
from torch.utils.data import Subset
from data_handling.cifar import RemappedSubset

# A) CelebA: Multi-label Random Flip
def _get_base_dataset_and_absolute_indices(dataset) -> Tuple[object, List[int]]:
    """
    Recursively extracts the underlying base dataset and the absolute indices
    of each sample in the original dataset.

    Supports arbitrary nesting: Subset(Subset(...(CelebA))).

    Args:
        dataset (Dataset or Subset): A possibly nested subset.

    Returns:
        Tuple[Dataset, List[int]]: The original dataset and the absolute indices.
    """
    if isinstance(dataset, Subset):
        base_dataset, base_indices = _get_base_dataset_and_absolute_indices(dataset.dataset)
        absolute_indices = [base_indices[i] for i in dataset.indices]
        return base_dataset, absolute_indices
    else:
        return dataset, list(range(len(dataset)))


def poison_labels_randomly(subset: Subset, flip_rate: float = 1.0) -> None:
    """
    Randomly flips multi-label attributes for CelebA in a non-targeted fashion.

    Each label bit has an independent probability `flip_rate` of being flipped (1 → 0 or 0 → 1).

    Args:
        subset (Subset): A client's local training subset (may be nested).
        flip_rate (float): Probability to flip each individual label bit. Default is 1.0.
    """
    base_dataset, absolute_indices = _get_base_dataset_and_absolute_indices(subset)
    print(f"[CelebA] Random label flip on {len(absolute_indices)} samples (flip_rate={flip_rate})")

    if not hasattr(base_dataset, "attr"):
        raise AttributeError(f"Dataset {type(base_dataset)} does not have the 'attr' attribute (expected CelebA).")

    for abs_idx in absolute_indices:
        original = base_dataset.attr[abs_idx].clone()
        flip_mask = torch.rand_like(original.float()) < flip_rate
        flipped = torch.remainder(original + flip_mask.long(), 2)
        base_dataset.attr[abs_idx] = flipped



# B) CIFAR-10: Single-label Random Flip
def get_targets_base_dataset_and_indices(dataset) -> Tuple[object, List[int]]:
    """
    Recursively extracts the base dataset that contains `.targets`
    and the corresponding absolute indices in that dataset.

    Supports nested structures such as: Subset(RemappedSubset(Subset(CIFAR10))).

    Args:
        dataset (Dataset or Subset): A dataset or nested subset.

    Returns:
        Tuple[Dataset, List[int]]: The dataset with `.targets` and absolute indices.

    Raises:
        AttributeError: If no dataset with `.targets` is found.
    """
    if isinstance(dataset, Subset):
        base, base_indices = get_targets_base_dataset_and_indices(dataset.dataset)
        absolute_indices = [base_indices[i] for i in dataset.indices]
        return base, absolute_indices
    elif isinstance(dataset, RemappedSubset):
        base, base_indices = get_targets_base_dataset_and_indices(dataset.subset)
        return base, base_indices
    elif hasattr(dataset, "targets"):
        return dataset, list(range(len(dataset)))
    else:
        raise AttributeError(f"No dataset with `.targets` found (current type: {type(dataset)}).")


def poison_labels_randomly_for_cifar(subset: Subset, num_classes: int = 10, flip_rate: float = 1.0) -> None:
    """
    Randomly flips single-label class values in CIFAR-10 by assigning a different class.

    For each sample, with probability `flip_rate`, the original label is replaced
    with a uniformly sampled incorrect class.

    Args:
        subset (Subset): A client's training subset (possibly nested).
        num_classes (int): Total number of classes (e.g., 6 for T1, 4 for T2).
        flip_rate (float): Probability of flipping the label for each sample.
    """
    base_dataset, absolute_indices = get_targets_base_dataset_and_indices(subset)
    print(f"[CIFAR] Random label flip on {len(absolute_indices)} samples (flip_rate={flip_rate}, K={num_classes})")

    for abs_idx in absolute_indices:
        if torch.rand(1).item() < flip_rate:
            orig = base_dataset.targets[abs_idx]
            new_label = torch.randint(0, num_classes, (1,)).item()
            while new_label == orig:
                new_label = torch.randint(0, num_classes, (1,)).item()
            base_dataset.targets[abs_idx] = new_label

# Make attack functions easily importable
__all__ = [
    "poison_labels_randomly",
    "poison_labels_randomly_for_cifar",
    "get_targets_base_dataset_and_indices",
]
