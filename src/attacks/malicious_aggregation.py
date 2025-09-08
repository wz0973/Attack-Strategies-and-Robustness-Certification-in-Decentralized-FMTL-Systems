# attacks/malicious_aggregation.py
import torch
"""
Malicious Aggregation Attacks.

This module implements post-aggregation model poisoning attacks that are injected
after backbone updates have been averaged (e.g., in HCA or FedPer).

Supported strategies:
- Gaussian Noise Injection
- Scaling / Sign-Flipping of Parameters
"""
def gaussian_noise_attack(encoders, noise_stddev=0.1):
    """
    Adds Gaussian noise to the aggregated encoder parameters for all clients.

    Args:
        encoders (List[Dict[str, torch.Tensor]]): A list of state_dicts representing the aggregated backbone for each client.
        noise_stddev (float): Standard deviation of the Gaussian noise to be added. Default is 0.1.

    Returns:
        List[Dict[str, torch.Tensor]]: Encoders with added Gaussian noise.
    """
    for idx in range(len(encoders)):
        encoder = encoders[idx]
        for key, param in encoder.items():
            noise = torch.randn_like(param) * noise_stddev
            encoder[key] = param + noise
    return encoders

def malicious_aggregation_attack(encoders, scale_factor=2.0):
    """
    Applies a malicious transformation (e.g., scaling or sign-flipping) to each client's aggregated backbone.

    This can simulate attacks such as:
    - Sign flipping (scale_factor = -1)
    - Boosting (scale_factor > 1)
    - Shrinking (scale_factor < 1)

    Args:
        encoders (List[Dict[str, torch.Tensor]]): A list of state_dicts representing each client’s averaged encoder.
        scale_factor (float): Factor to scale each parameter. Negative values perform sign-flip. Default is 2.0.

    Returns:
        List[Dict[str, torch.Tensor]]: Encoders with maliciously scaled parameters.
    """
    for idx in range(len(encoders)):
        encoder = encoders[idx]
        for key, param in encoder.items():
            encoder[key] = param * scale_factor
            if torch.isnan(encoder[key]).any():
                print(f"NaN detected in encoder {idx}, layer {key} after scaling.")
    return encoders
