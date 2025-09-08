"""
Scaled Boost Attack on Model Updates.

This module implements a model poisoning strategy that amplifies the local update
(ΔW = W_t - W_{t-1}) before submitting it for aggregation.

The attack is intended to disproportionately influence the global model by scaling
the difference between the current and previous backbone states.
"""

import torch
def scale_model_update(prev_backbone, curr_backbone, scale_factor=5.0):
    """
    Amplifies the model update by scaling the delta between current and previous backbone.

    Args:
        prev_backbone (dict): Previous round backbone state dict.
        curr_backbone (dict): Current round backbone state dict.
        scale_factor (float): Factor to scale the update (ΔW). Default is 5.0.

    Returns:
        dict: A new state dict with scaled parameters.
    """
    if prev_backbone is None or curr_backbone is None:
        raise ValueError("Backbones are not ready.")

    bn_kw = ("bn", "downsample.1", "running_mean", "running_var")
    scaled = {}

    for k in curr_backbone:
        # Skip batchnorm and downsample stats
        if any(word in k for word in bn_kw):
            scaled[k] = curr_backbone[k].clone()
            continue

        delta = curr_backbone[k] - prev_backbone[k]
        # Sanitize possible numerical issues
        if torch.isnan(delta).any() or torch.isinf(delta).any():
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        s_param = prev_backbone[k] + scale_factor * delta
        s_param = torch.clamp(s_param, -1000.0, 1000.0)
        s_param = torch.nan_to_num(s_param, nan=0.0, posinf=0.0, neginf=0.0)
        scaled[k] = s_param
    return scaled


def update_model_with_backbone(model, device, backbone_dict):
    """
    Updates the model's backbone with a custom (e.g., poisoned) state dict.

    Args:
        model (torch.nn.Module): The model to update.
        device (torch.device): The target device (e.g., cuda or cpu).
        backbone_dict (dict): A state_dict for the model's backbone.
    """
    # Strip "backbone." prefix and sanitize values
    cleaned = {k.replace("backbone.", ""): v for k, v in backbone_dict.items()}
    for k, v in cleaned.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            cleaned[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    model.backbone.load_state_dict(cleaned, strict=True)
    model.to(device)


def reestimate_bn(model, loader, device, n_batches=50):
    """
    Re-estimates BatchNorm statistics by running forward passes on training data.

    This ensures consistency after major parameter perturbations.

    Args:
        model (torch.nn.Module): The model to update.
        loader (DataLoader): The data loader used for BN forward pass.
        device (torch.device): Target device.
        n_batches (int): Number of batches to use. Default is 50.
    """
    model.train()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= n_batches:
                break
            _ = model(inputs.to(device))
    model.eval()


def apply_scaled_boost_attack(client, scale_factor=5.0):
    """
    Applies the scaled boost attack on a single client.

    This wrapper function:
    - Scales the update between prev and curr backbones
    - Updates the model accordingly
    - Re-estimates BN statistics
    - Overwrites the current backbone with the poisoned version

    Args:
        client (Client): A client object with prev/curr backbones and model.
        scale_factor (float): Amplification factor for the update. Default is 5.0.
    """
    scaled = scale_model_update(client.prev_backbone, client.curr_backbone, scale_factor)
    update_model_with_backbone(client.model, client.device, scaled)
    reestimate_bn(client.model, client.model.train_loader, client.device)
    client.curr_backbone = scaled
    client.update_checkpoint(client.current_checkpoint["epoch"])
