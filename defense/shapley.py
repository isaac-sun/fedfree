"""
Monte Carlo per-class Shapley value estimation.

Per-class SV:  SV_{i,c} = marginal contribution of client i to the
cross-entropy loss of class c on the validation set.

Reference: SVRFL (Zhu et al.)
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.lora_model import create_model, load_lora_state_dict


def _build_coalition_params(
    global_state_dict: OrderedDict,
    updates: dict[int, OrderedDict],
    coalition: list[int],
    sample_counts: dict[int, int],
) -> OrderedDict:
    """Build coalition model params:  w_S = w_g + Σ n_j·Δ_j / Σ n_j."""
    if len(coalition) == 0:
        return OrderedDict({k: v.clone() for k, v in global_state_dict.items()})

    total_n = sum(sample_counts.get(cid, 1) for cid in coalition) or 1
    ref_keys = list(updates[next(iter(coalition))].keys())
    new_state = OrderedDict()

    for key in ref_keys:
        agg = torch.zeros_like(updates[coalition[0]][key], dtype=torch.float64,
                               device=global_state_dict[key].device)
        for cid in coalition:
            n_j = sample_counts.get(cid, 1)
            agg = agg + n_j * updates[cid][key].to(dtype=torch.float64, device=agg.device)
        new_state[key] = (
            global_state_dict[key].to(dtype=torch.float64, device=global_state_dict[key].device) + agg / total_n
        ).to(dtype=global_state_dict[key].dtype)
    return new_state


@torch.no_grad()
def _evaluate_per_class_loss(
    model_config: dict,
    state_dict: OrderedDict,
    data_loader: DataLoader,
    num_classes: int,
    device: str,
) -> np.ndarray:
    """Evaluate per-class average cross-entropy loss.

    Returns ndarray of shape (num_classes,).
    """
    model = create_model(
        model_name=model_config["model_name"],
        num_classes=num_classes,
        lora_r=model_config["lora_r"],
        lora_alpha=model_config["lora_alpha"],
        lora_dropout=model_config["lora_dropout"],
        lora_target_modules=model_config["lora_target_modules"],
    )
    load_lora_state_dict(model, state_dict)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="none")
    class_loss_sum = np.zeros(num_classes, dtype=np.float64)
    class_count = np.zeros(num_classes, dtype=np.int64)

    for batch in data_loader:
        input_ids, attn_mask, labels = [b.to(device) for b in batch]
        per_sample_loss = criterion(model(input_ids, attn_mask), labels)
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                class_loss_sum[c] += per_sample_loss[mask].sum().item()
                class_count[c] += mask.sum().item()

    safe_count = np.maximum(class_count, 1)
    return class_loss_sum / safe_count


def estimate_round_shapley_per_class(
    model_config: dict,
    updates: dict[int, OrderedDict],
    global_state_dict: OrderedDict,
    val_loader: DataLoader,
    num_classes: int,
    num_mc_samples: int = 10,
    sample_counts: dict[int, int] | None = None,
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    """Estimate per-class Shapley values for one FL round via Monte Carlo.

    Returns dict: client_id -> np.ndarray of shape (num_classes,)
    """
    client_ids = list(updates.keys())
    if len(client_ids) == 0:
        return {}

    if sample_counts is None:
        sample_counts = {cid: 1 for cid in client_ids}

    shapley_sums = {cid: np.zeros(num_classes, dtype=np.float64) for cid in client_ids}

    base_pc = _evaluate_per_class_loss(
        model_config, global_state_dict, val_loader, num_classes, device
    )

    for _ in range(num_mc_samples):
        perm = np.random.permutation(client_ids).tolist()
        coalition = []
        prev_pc = base_pc.copy()

        for cid in perm:
            coalition.append(cid)
            coal_params = _build_coalition_params(
                global_state_dict, updates, coalition, sample_counts
            )
            curr_pc = _evaluate_per_class_loss(
                model_config, coal_params, val_loader, num_classes, device
            )
            shapley_sums[cid] += prev_pc - curr_pc
            prev_pc = curr_pc

    return {cid: shapley_sums[cid] / num_mc_samples for cid in client_ids}


def per_class_to_overall(
    per_class_sv: dict[int, np.ndarray],
    class_weights: np.ndarray,
) -> dict[int, float]:
    """Weighted sum of per-class SV → overall SV."""
    return {cid: float(np.dot(class_weights, sv)) for cid, sv in per_class_sv.items()}


def _class_weights_from_loader(val_loader, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for batch in val_loader:
        labels = batch[2].numpy()
        for c in range(num_classes):
            counts[c] += (labels == c).sum()
    total = counts.sum()
    return counts / max(total, 1)
