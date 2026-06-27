"""FedAvg aggregation with optional server momentum."""

from __future__ import annotations

from collections import OrderedDict

import torch


def fedavg_aggregate(
    global_state_dict: OrderedDict,
    updates: dict[int, OrderedDict],
    server_lr: float = 1.0,
    weights: dict[int, float] | None = None,
    momentum: float = 0.0,
    momentum_buffer: OrderedDict | None = None,
) -> tuple[OrderedDict, OrderedDict | None]:
    """FedAvg:  w_new = w_g + server_lr * weighted_avg(updates)."""
    client_ids = list(updates.keys())
    n = len(client_ids)

    if n == 0:
        return (
            OrderedDict({k: v.clone() for k, v in global_state_dict.items()}),
            momentum_buffer,
        )

    if weights is None:
        weights = {cid: 1.0 / n for cid in client_ids}

    new_state = OrderedDict()
    new_momentum = OrderedDict() if momentum > 0 else None

    for key in global_state_dict:
        device = global_state_dict[key].device
        dtype = global_state_dict[key].dtype

        avg_delta = sum(
            weights[cid] * updates[cid][key].to(device) for cid in client_ids
        )

        if momentum > 0 and momentum_buffer is not None and key in momentum_buffer:
            avg_delta = momentum * momentum_buffer[key].to(device) + avg_delta

        if new_momentum is not None:
            new_momentum[key] = avg_delta.clone()

        new_state[key] = global_state_dict[key] + server_lr * avg_delta

    return new_state, new_momentum
