"""Free-rider attack implementations.

Three attack types from the SVRFL paper (Zhu et al.):
- DFR  (Disguised Free-Rider)    — Fraboni et al.: σ·t^(-γ)·N(0,I)
- SDFR (Scaled Delta Free-Rider)  — Zhu et al.: ||Δt||/||Δt-1||·Δt
- AFR  (Advanced Free-Rider)     — Zhu et al.: SDFR + calibrated sparse noise
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch


# ═══════════════════════════════════════════════════════════════════════════════
# DFR — Disguised Free-Rider (Fraboni et al.)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_dfr_sigma(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
) -> Optional[float]:
    """Estimate DFR sigma from the first observed global model delta."""
    if len(global_history) < 1:
        return None

    w_init = global_history[0]
    w_after_first = global_history[1] if len(global_history) >= 2 else global_state_dict
    delta_flat = torch.cat([
        (w_after_first[k].float() - w_init[k].float()).flatten()
        for k in w_init.keys()
    ])
    return delta_flat.std().item()


def dfr_attack(
    global_state_dict: OrderedDict,
    sigma: float = 0.5,
    round_num: int = 1,
    gamma: float = 1.0,
) -> OrderedDict:
    """Disguised Free-Rider — Gaussian noise with round-decaying variance.

    Formula:  update = σ · t^(-γ) · N(0, I)

    Parameters
    ----------
    sigma : standard deviation at round 1.
    round_num : current FL round (1-indexed).
    gamma : decay exponent.
    """
    phi_t = sigma * (max(round_num, 1) ** (-gamma))
    update = OrderedDict()
    for key, val in global_state_dict.items():
        update[key] = torch.randn_like(val) * phi_t
    return update


# ═══════════════════════════════════════════════════════════════════════════════
# SDFR — Scaled Delta Free-Rider (Zhu et al.)
# ═══════════════════════════════════════════════════════════════════════════════

def sdfr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    eps: float = 1e-10,
) -> OrderedDict:
    """Scaled Delta Free-Rider — copies global delta direction with scaling.

    Formula:  U_f = ||Δt|| / ||Δt-1|| · Δt    where Δt = θ(t) − θ(t-1)

    Edge cases:
    - k=0 (no history):      returns small noise
    - k=1 (one prior state): returns raw delta θ(t) − θ(t-1)
    - k≥2:                   full scaling formula
    """
    keys = list(global_state_dict.keys())
    k = len(global_history)

    if k == 0:
        update = OrderedDict()
        for key in keys:
            update[key] = torch.randn_like(global_state_dict[key]) * 1e-4
        return update

    if k == 1:
        w_prev = global_history[-1]
        update = OrderedDict()
        for key in keys:
            update[key] = global_state_dict[key] - w_prev[key]
        return update

    # Full formula (k ≥ 2)
    w_t = global_state_dict
    w_t1 = global_history[-1]
    w_t2 = global_history[-2]

    delta_t_norm_sq = 0.0
    delta_prev_norm_sq = 0.0
    delta_t = OrderedDict()

    for key in keys:
        dt = w_t[key].float() - w_t1[key].float()
        dp = w_t1[key].float() - w_t2[key].float()
        delta_t[key] = w_t[key] - w_t1[key]
        delta_t_norm_sq += dt.pow(2).sum().item()
        delta_prev_norm_sq += dp.pow(2).sum().item()

    scale = math.sqrt(delta_t_norm_sq) / (math.sqrt(delta_prev_norm_sq) + eps)

    update = OrderedDict()
    for key in delta_t:
        update[key] = delta_t[key] * scale

    return update


# ═══════════════════════════════════════════════════════════════════════════════
# AFR — Advanced Free-Rider (Zhu et al.)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_e_cos_beta(val_loss_t: float, val_loss_init: float) -> float:
    """Estimate E[cos β] from the validation-loss trajectory.

    From Zhu et al.:  E[cos β](t) = 1 − l(t) / l(1)

    - At round 1:    l(t) ≈ l(1) → E[cos β] ≈ 0  (uncorrelated updates)
    - Converged:     l(t) → 0   → E[cos β] → 1    (highly aligned updates)
    """
    if val_loss_init is None or val_loss_init < 1e-10:
        return 0.0
    ratio = val_loss_t / val_loss_init
    return max(0.0, min(1.0 - ratio, 0.99))


class AFRState:
    """Track running statistics for paper-faithful AFR noise calibration.

    Maintains:
    1. val_loss_init — initial validation loss l(1) for E[cos β] estimation.
    2. base_norm_ema  — EMA of ||U_f(θ)||, approximating |E[U_f(θ)]|.
    """

    def __init__(self, ema_alpha: float = 0.3):
        self.val_loss_init: Optional[float] = None
        self.base_norm_ema: Optional[float] = None
        self.ema_alpha = ema_alpha

    def update(self, val_loss_t: float, base_norm_t: float):
        """Update tracked quantities after each round."""
        if self.val_loss_init is None:
            self.val_loss_init = val_loss_t
        a = self.ema_alpha
        if self.base_norm_ema is None:
            self.base_norm_ema = base_norm_t
        else:
            self.base_norm_ema = a * base_norm_t + (1 - a) * self.base_norm_ema

    def get_e_cos_beta(self, val_loss_t: float) -> float:
        """Estimate E[cos β] from current-round validation loss."""
        return estimate_e_cos_beta(val_loss_t, self.val_loss_init)

    def get_mean_base_norm(self) -> Optional[float]:
        """Return EMA of ||U_f(θ)||, or None if no history yet."""
        return self.base_norm_ema


def afr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    n_total: int = 10,
    e_cos_beta: float = 0.0,
    mean_base_norm: Optional[float] = None,
    noisy_frac: float = 0.1,
    seed: Optional[int] = None,
    eps: float = 1e-10,
) -> Tuple[OrderedDict, float]:
    """Advanced Free-Rider — SDFR + calibrated sparse Gaussian noise.

    Extends SDFR by adding sparse noise whose magnitude |φ(t)| is
    calibrated so that the cosine similarity between the AFR update
    and honest updates stays plausible.

    Paper formula:
        |φ(t)| = sqrt( n² / (n + (n²−n)·E[cos β]) − 1 ) · |E[U_f(θ)]|

    Noise is applied to d = noisy_frac · D randomly selected coordinates:
        z_i ~ N(0, φ(t)² / d)   for selected coords;   0 elsewhere
    This yields ||z|| ≈ |φ(t)| regardless of d.

    Parameters
    ----------
    n_total : total number of clients in the FL system.
    e_cos_beta : estimated E[cos β] (from AFRState or direct override).
    mean_base_norm : EMA of past SDFR output norms; if None, uses current ||U_f||.
    noisy_frac : fraction of parameters to perturb (d / D).

    Returns
    -------
    update : the AFR fake update.
    base_norm : ||U_f(θ)|| for this round (caller feeds into AFRState.update).
    """
    # Step 1: SDFR base update
    base_update = sdfr_attack(global_state_dict, global_history, eps=eps)

    # Step 2: flatten into a single vector
    keys = list(base_update.keys())
    shapes = [base_update[k].shape for k in keys]
    flat_base = torch.cat([base_update[k].flatten() for k in keys])
    total_dim = flat_base.numel()  # D

    # Step 3: compute norms
    base_norm = flat_base.float().norm().item()  # ||U_f(θ)|| this round
    effective_base_norm = mean_base_norm if mean_base_norm is not None else base_norm

    # Step 4: compute noise magnitude |φ(t)|
    n = max(n_total, 2)
    denom = n + (n * n - n) * e_cos_beta
    ratio = (n * n) / (denom + eps)
    phi_t_sq = max(ratio - 1.0, 0.0) * (effective_base_norm ** 2)
    phi_t = math.sqrt(phi_t_sq)

    # Step 5: select d random coordinates
    num_noisy = max(int(total_dim * noisy_frac), 1)  # d

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.manual_seed(torch.randint(0, 2**31, (1,)).item())

    indices = torch.randperm(total_dim, generator=gen)[:num_noisy]

    # Step 6: sparse noise z ~ N(0, φ²/d) on selected coords
    noise_std = phi_t / math.sqrt(num_noisy)
    noise_values = torch.randn(num_noisy, generator=gen).to(
        dtype=flat_base.dtype, device=flat_base.device
    ) * noise_std

    flat_noise = torch.zeros_like(flat_base)
    flat_noise[indices] = noise_values

    # Step 7: reconstruct update dict
    flat_result = flat_base + flat_noise

    update = OrderedDict()
    offset = 0
    for k, shape in zip(keys, shapes):
        numel = 1
        for s in shape:
            numel *= s
        update[k] = flat_result[offset:offset + numel].reshape(shape)
        offset += numel

    return update, base_norm


# ═══════════════════════════════════════════════════════════════════════════════
# Backward-compatible aliases
# ═══════════════════════════════════════════════════════════════════════════════

def noise_attack(
    global_state_dict: OrderedDict,
    sigma: float = 0.1,
    device: str = "cpu",
) -> OrderedDict:
    """Legacy alias — use dfr_attack() instead."""
    return dfr_attack(global_state_dict, sigma=sigma)


def disguise_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    device: str = "cpu",
    eps: float = 1e-10,
) -> OrderedDict:
    """Legacy alias — use sdfr_attack() instead."""
    return sdfr_attack(global_state_dict, global_history, eps=eps)
