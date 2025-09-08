"""
Sign Flip Attack on Backbone Updates.

This module implements a simple yet effective model poisoning strategy
that reverses the direction of the model update (ΔW), causing harmful aggregation
during federated learning.

The attack computes:
    flipped = prev_backbone - (curr_backbone - prev_backbone)
            = 2 * prev_backbone - curr_backbone
"""

import torch
def flip_model_update(prev_backbone: dict, curr_backbone: dict) -> dict:
    """
    Applies the Sign Flip Attack to reverse the direction of the update.

    Args:
        prev_backbone (dict): State dict of the model from the previous round.
        curr_backbone (dict): State dict of the model from the current round.

    Returns:
        dict: The maliciously flipped backbone.
    """

    flipped = {}

    for key in curr_backbone:
        delta = curr_backbone[key] - prev_backbone[key]
        flipped[key] = prev_backbone[key] - delta   # Equivalent to 2*prev - curr

    print("Sign Flip Attack applied to backbone.")
    return flipped
