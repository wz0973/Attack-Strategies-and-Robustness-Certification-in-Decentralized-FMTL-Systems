import numpy as np
import torch
from scipy.optimize import minimize

"""
This script implements a simplified version of the Hyper Conflict Averse aggregation mechanism proposed by:
Lu, Y., Huang, S., Yang, Y., Sirejiding, S., Ding, Y., & Lu, H. (2024). 
FedHCA2: Towards Hetero-Client Federated Multi-Task Learning. 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5599-5609).

The functions below are a subset of functions developed by the authors above. Only minor modifications have taken place.
As such credits of the code below goes to the authors above!
Their source code is accesible here: https://github.com/innovator-zero/FedHCA2
"""


def get_delta_dict_list(param_dict_list, last_param_dict_list):
    """
    Get the difference between current and last parameters
    """
    # a list of length N, each element is a dict of delta parameters
    delta_dict_list = []
    layers = param_dict_list[0].keys()
    for i in range(len(param_dict_list)):
        delta_dict_list.append({})
        for layer in layers:
            delta_dict_list[i][layer] = (
                param_dict_list[i][layer] - last_param_dict_list[i][layer]
            )

    return delta_dict_list


def new_get_encoder_params(all_clients):
    """
    Get encoder parameters from each client's current checkpoint. (Note: Encoder translates to backbone as part of this project.)
    """
    # Assuming that 'current_checkpoint' contains a dictionary with 'model_state_dict'
    first_client_state_dict = all_clients[0]
    all_name_keys = list(first_client_state_dict.keys())

    encoder_param_dict_list = []
    layers = []
    shapes = []

    for client in all_clients:
        param_dict = {}
        model_state_dict = client
        for key in all_name_keys:
            prefix, layer = key.split(".", 1)
            param_dict[layer] = model_state_dict[key]
        encoder_param_dict_list.append(param_dict)

    # Get layers and shapes (same for all encoders)
    for key in all_name_keys:
        layer = key.split(".", 1)[1]
        layers.append(layer)
        shapes.append(model_state_dict[key].shape)

    return encoder_param_dict_list, all_name_keys, layers, shapes


def get_ca_delta(flatten_delta_list, alpha, rescale=1):
    """
    Solve for aggregated conflict-averse delta

    Args:
        flatten_delta_list ([]): A list of the flatted deltas 把每个参数 dict 变成一维向量
        alpha (float): Is a scaling factor that controls the influence of the norm of the gradients
            on the objective function during the optimization process, helping to balance the magnitude of the update.
        rescale (int, optional): Modifies the impact of alpha on the update.

    Returns:
        final_update: The calculated hyper conflict averse update.
    """

    print(alpha)
    N = len(flatten_delta_list)
    grads = torch.stack(flatten_delta_list).t()  # [d , N]
    GG = grads.t().mm(grads).cpu()  # [N, N]
    assert not torch.isnan(GG).any(), "NaN detected in GG matrix"
    g0_norm = (GG.mean() + 1e-8).sqrt()

    x_start = np.ones(N) / N
    bnds = tuple((0, 1) for x in x_start)
    cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
    A = GG.numpy()
    assert not torch.isnan(GG).any(), "NaN detected in GG matrix"
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
            + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    ww = torch.Tensor(res.x).to(grads.device)
    assert not torch.isnan(ww).any(), "NaN detected in optimization result"

    gw = (grads * ww.reshape(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        final_update = g
    elif rescale == 1:
        final_update = g / (1 + alpha**2)
    else:
        final_update = g / (1 + alpha)

    return final_update


def flatten_param(param_dict_list, layers):
    """
    Flattens a list of parameter dictionaries into a list of 1D tensors.
    """

    flatten_list = [
        torch.cat([param_dict_list[idx][layer].flatten() for layer in layers])
        for idx in range(len(param_dict_list))
    ]
    assert len(flatten_list[0].shape) == 1

    return flatten_list


def unflatten_param(flatten_list, shapes, layers):
    """
    Reconstructs a list of parameter dictionaries from flattened tensors.
    """

    param_dict_list = []
    for model_idx in range(len(flatten_list)):
        start = 0
        param_dict_list.append({})
        for layer, shape in zip(layers, shapes):
            end = start + int(np.prod(shape))
            param_dict_list[model_idx][layer] = flatten_list[model_idx][
                start:end
            ].reshape(shape)
            start = end

    return param_dict_list


def conflict_averse(curr_backbones_dicts, prev_backbones_dicts, ca_c, title_of_experiment=""):

    """
    Aggregates model parameters using a conflict-averse update strategy.

    Args:
        curr_backbones_dicts ([dict]): A list of dictionaries. Each contains the current model parameters.
        prev_backbones_dicts ([dict]): A list of dictionaries. Each contains the previous model parameters.
        ca_c (float): Conflict-averse hyperparameter that controls the impact of the aggregated update to reduce
                      conflicts between the current and previous model states.

    Returns:
        [{}]: Returns a list of updated parameter dictionaries for each client.
    """

    N = len(curr_backbones_dicts)

    # update_ckpt = copy.deepcopy(save_ckpt)  # store updated parameters

    # Get encoder parameter list
    encoder_param_list, encoder_keys, enc_layers, enc_shapes = new_get_encoder_params(
        curr_backbones_dicts
    )

    # Encoder agg
    last_encoder_param_list, _, _, _ = new_get_encoder_params(prev_backbones_dicts)
    encoder_delta_list = get_delta_dict_list(
        encoder_param_list, last_encoder_param_list
    )

    # Flatten
    flatten_last_encoder = flatten_param(last_encoder_param_list, enc_layers)
    del last_encoder_param_list
    flatten_encoder_delta = flatten_param(encoder_delta_list, enc_layers)
    del encoder_delta_list

    # Solve for aggregated conflict-averse delta
    flatten_delta_update = get_ca_delta(flatten_encoder_delta, ca_c)  # flattened tensor

    for idx, client_encoder in enumerate(flatten_last_encoder):
        client_encoder.add_(flatten_encoder_delta[idx] + 1 * flatten_delta_update)
    flatten_new_encoders = flatten_last_encoder

    new_encoders = unflatten_param(flatten_new_encoders, enc_shapes, enc_layers)

    from attacks.malicious_aggregation import malicious_aggregation_attack
    if "POISON_Malicious_Aggregation" in title_of_experiment:
        print("🎯 Injecting malicious aggregation attack")
        new_encoders = malicious_aggregation_attack(new_encoders, scale_factor=2.0)

    for encoder in new_encoders:
        # Update the dictionary with new keys in place
        for key in list(encoder.keys()):
            encoder[f"backbone.{key}"] = encoder.pop(key)

    return new_encoders