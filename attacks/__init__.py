"""Free-rider attack implementations."""

from __future__ import annotations

from collections import OrderedDict

import torch


def noise_attack(
    global_state_dict: OrderedDict,
    sigma: float = 0.1,
    device: str = "cpu",
) -> OrderedDict:
    """Trivial free-rider: sends Gaussian noise.

    This simulates the simplest free-rider who contributes nothing useful
    but generates random updates to avoid being trivially detected as
    'zero-update' clients.
    """
    update = OrderedDict()
    for key, val in global_state_dict.items():
        update[key] = torch.randn_like(val, device=device) * sigma
    return update


def disguise_attack(
    global_state_dict: OrderedDict,
    global_history: list[OrderedDict],
    device: str = "cpu",
    eps: float = 1e-10,
) -> OrderedDict:
    """Disguised free-rider: copies global delta direction with scaling.

    This is a simplified SDFR variant: the attacker computes the direction
    of the most recent global model change and submits a scaled copy,
    mimicking honest participation while contributing nothing novel.
    """
    if len(global_history) < 2:
        # Not enough history — fall back to noise
        return noise_attack(global_state_dict, sigma=1e-3, device=device)

    w_t = global_state_dict
    w_t1 = global_history[-1]
    w_t2 = global_history[-2]

    delta_norm_sq = 0.0
    prev_norm_sq = 0.0
    delta_t = OrderedDict()

    for key in global_state_dict:
        dt = w_t[key].float() - w_t1[key].float()
        dp = w_t1[key].float() - w_t2[key].float()
        delta_t[key] = (w_t[key] - w_t1[key]).to(device)
        delta_norm_sq += dt.pow(2).sum().item()
        prev_norm_sq += dp.pow(2).sum().item()

    scale = (delta_norm_sq ** 0.5) / (prev_norm_sq ** 0.5 + eps)

    update = OrderedDict()
    for key in delta_t:
        update[key] = delta_t[key].to(device) * scale

    return update
