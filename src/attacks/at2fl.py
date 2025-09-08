import torch
import torch.nn as nn
"""
AT2FL attack implementation.
Supports multi-label classification (CelebA) and single-label classification (CIFAR-10).
"""

def at2fl_attack_celeba(model, epsilon=0.015, poison_batches=8, reestimate_bn=True):
    """
    Performs AT2FL attack on a CelebA model.

    This variant assumes input range is [-1, 1] and applies weight clipping to ±5.

    Args:
        model (nn.Module): The model to be poisoned.
        epsilon (float): Perturbation strength for adversarial examples.
        poison_batches (int): Number of batches to poison.
        reestimate_bn (bool): Whether to re-estimate batch norm statistics after attack.
    """
    _at2fl_shared_logic(model, epsilon, poison_batches, reestimate_bn, clamp_min=-1.0, clamp_max=1.0, weight_clip=5.0)


def at2fl_attack_cifar(model, epsilon=0.03, poison_batches=10, reestimate_bn=True):
    """
    Performs AT2FL attack on a CIFAR-10 model.

    This variant assumes input range is [0, 1] and applies weight clipping to ±10.

    Args:
        model (nn.Module): The model to be poisoned.
        epsilon (float): Perturbation strength for adversarial examples.
        poison_batches (int): Number of batches to poison.
        reestimate_bn (bool): Whether to re-estimate batch norm statistics after attack.
    """
    _at2fl_shared_logic(model, epsilon, poison_batches, reestimate_bn, clamp_min=0.0, clamp_max=1.0, weight_clip=10.0)

def _at2fl_shared_logic(model, epsilon, poison_batches, reestimate_bn, clamp_min, clamp_max, weight_clip):
    """
        Core logic for AT2FL adversarial attack across different datasets.

        Steps:
        1. For N batches, generate adversarial inputs using FGSM: x_adv = x + ε * sign(∇xL).
        2. Train the model on these adversarial samples.
        3. Clamp backbone weights and sanitize parameters to remove NaNs/Infs.
        4. Optionally re-estimate BatchNorm statistics.

        Args:
            model (nn.Module): The model to attack.
            epsilon (float): Magnitude of input perturbation.
            poison_batches (int): How many batches to poison.
            reestimate_bn (bool): If True, re-run data through model to refresh BN stats.
            clamp_min (float): Minimum value after perturbation (input range lower bound).
            clamp_max (float): Maximum value after perturbation (input range upper bound).
            weight_clip (float): Max absolute value for backbone weight clipping.
        """
    device = model.device
    model.train()
    train_loader = model.train_loader

    for b_idx, (inputs, labels) in enumerate(train_loader):
        if 0 <= poison_batches <= b_idx:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad_(True)

        # Step 1: Compute input gradient ∇xL
        model.optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = model.criterion(outputs, labels)
        loss.backward()
        # Step 2: Generate adversarial input using FGSM
        grad_sign = inputs.grad.sign()
        adv_inputs = torch.clamp(inputs + epsilon * grad_sign, clamp_min, clamp_max).detach()
        # Skip poisoned batch if invalid values are introduced
        if torch.isnan(adv_inputs).any() or torch.isinf(adv_inputs).any():
            print("⚠️ Adversarial inputs contain NaN/Inf — skipping batch.")
            continue
        # Step 3: Backprop through adversarial inputs
        model.optimizer.zero_grad(set_to_none=True)
        adv_outputs = model(adv_inputs)
        adv_loss = model.criterion(adv_outputs, labels)
        adv_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.optimizer.step()

    # Step 4: Sanitize parameters (remove NaNs, Infs)
    with torch.no_grad():
        for p in model.parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)

        for b in model.buffers():
            if torch.isnan(b).any() or torch.isinf(b).any():
                b.data = torch.nan_to_num(b.data, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip backbone weights
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.data.clamp_(-weight_clip, weight_clip)
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f" Cleaning parameter: {name}")
                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=weight_clip, neginf=-weight_clip)

    # Step 5: Optional BatchNorm re-estimation
    if reestimate_bn and hasattr(model, "train_loader"):
        model.eval()
        for i, (bn_x, _) in enumerate(model.train_loader):
            if i >= min(20, len(model.train_loader)):
                break
            with torch.no_grad():
                _ = model(bn_x.to(device))
        model.train()
    # Mark attack as performed
    model.at2fl_enabled = True
