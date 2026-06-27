#!/usr/bin/env python3
"""
Federated Learning Free-Rider Defense — Main Training Script.

Runs two experiments:
  1.  FedAvg baseline (no defense)
  2.  FedAvg + DefenseController (two-phase per-class Shapley detection)

Attackers: 40% of clients are free-riders (noise or disguise).

Outputs:
  - results/defense_history.csv   — per-round SV metrics for visualization
  - Console summary table
"""

from __future__ import annotations

import os
import sys
import copy
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config
from utils.seed import set_seed
from utils.metrics import evaluate_model
from utils.export import export_defense_csv
from data.yahoo_answers import load_yahoo_answers
from models.lora_model import create_model, get_lora_state_dict, load_lora_state_dict
from fl.client import FLClient
from fl.server import FLServer
from fl.fedavg import fedavg_aggregate
from attacks import noise_attack, disguise_attack
from defense.shapley import (
    estimate_round_shapley_per_class,
    per_class_to_overall,
    _class_weights_from_loader,
)
from defense.controller import DefenseController


def _stamp(msg: str):
    print(f"[fedfree] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Single Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    config: Config,
    train_ds,
    val_ds,
    test_ds,
    class_names: list[str],
    defense_controller: DefenseController | None = None,
):
    """Run one FL experiment.

    Parameters
    ----------
    defense_controller : DefenseController or None
        If None, runs vanilla FedAvg.  If provided, runs with defense.
    """
    set_seed(config.seed)
    tag = "Defense" if defense_controller else "Baseline"
    _stamp(f"=== Experiment: {config.attack_type} | {tag} ===")

    # ── Partition data across clients ──────────────────────────────────────
    n = len(train_ds)
    indices = np.random.permutation(n)
    chunk_size = n // config.num_clients
    partition = {}
    for cid in range(config.num_clients):
        start = cid * chunk_size
        end = start + chunk_size if cid < config.num_clients - 1 else n
        partition[cid] = indices[start:end]
    client_datasets = {cid: Subset(train_ds, idxs) for cid, idxs in partition.items()}

    # ── Model ──────────────────────────────────────────────────────────────
    model_config = {
        "model_name": config.model_name,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target_modules": config.lora_target_modules,
    }

    model = create_model(
        num_classes=config.num_classes, **model_config
    )
    server = FLServer(model, val_ds, test_ds, config)
    clients = {
        cid: FLClient(cid, client_datasets[cid], config)
        for cid in range(config.num_clients)
    }

    # ── Attackers ──────────────────────────────────────────────────────────
    num_mal = int(config.num_clients * config.malicious_ratio) if config.attack_type != "none" else 0
    attacker_ids = set(range(num_mal))

    # ── Shapley setup ──────────────────────────────────────────────────────
    eval_config = copy.deepcopy(config)
    eval_config.batch_size = config.eval_batch_size
    val_loader = DataLoader(val_ds, batch_size=config.eval_batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.eval_batch_size)
    class_weights = _class_weights_from_loader(val_loader, config.num_classes)

    # ── Tracking ───────────────────────────────────────────────────────────
    test_f1s = []
    summary_rows = []

    # ── FL Rounds ──────────────────────────────────────────────────────────
    for round_t in tqdm(range(config.num_rounds), desc=f"{tag} {config.attack_type}"):
        selected = server.select_clients(config.num_clients, config.participation_ratio)
        global_sd = server.get_global_state_dict()

        # Collect client updates
        updates = {}
        for cid in selected:
            if cid in attacker_ids:
                if config.attack_type == "noise":
                    updates[cid] = noise_attack(global_sd, sigma=0.1, device=config.device)
                elif config.attack_type == "disguise":
                    updates[cid] = disguise_attack(
                        global_sd, list(server.global_history), device=config.device
                    )
                else:
                    updates[cid] = clients[cid].train(global_sd)
            else:
                updates[cid] = clients[cid].train(global_sd)

        # ── Per-class Shapley estimation ──────────────────────────────────
        per_class_sv = estimate_round_shapley_per_class(
            model_config=model_config,
            updates=updates,
            global_state_dict=global_sd,
            val_loader=val_loader,
            num_classes=config.num_classes,
            num_mc_samples=config.num_mc_samples,
            device=config.device,
        )
        shapley_vals = per_class_to_overall(per_class_sv, class_weights)

        # ── Defense detection ──────────────────────────────────────────────
        if defense_controller is not None:
            detection = defense_controller.detect(
                per_class_sv=per_class_sv,
                client_ids=selected,
                attacker_ids=attacker_ids,
                round_num=round_t,
            )
            # Remove suspected clients from aggregation
            honest_updates = {
                cid: upd for cid, upd in updates.items()
                if not detection.get(cid, {}).get("suspected", False)
            }
            if len(honest_updates) == 0:
                honest_updates = updates  # fallback: aggregate everyone
        else:
            honest_updates = updates

        # ── FedAvg aggregation ────────────────────────────────────────────
        new_sd, _ = fedavg_aggregate(global_sd, honest_updates, server_lr=config.server_lr)
        server.update_global_model(new_sd)

        # ── Evaluation ────────────────────────────────────────────────────
        loss, macro_f1 = server.evaluate()
        test_f1s.append(macro_f1)

        if round_t % 5 == 0 or round_t == config.num_rounds - 1:
            _stamp(f"  Round {round_t:>2d}: macro_f1={macro_f1:.4f}  loss={loss:.4f}")

    # ── Final summary ─────────────────────────────────────────────────────
    final_f1 = test_f1s[-1]
    _stamp(f"  Final: macro_f1={final_f1:.4f}")
    return test_f1s


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _stamp("Python bootstrap starting")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _stamp(f"  Device: {device}")
    if device == "cuda":
        _stamp(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_ds, val_ds, test_ds, class_names = load_yahoo_answers(
        max_seq_length=256,   # Yahoo answers are shorter
        max_train=20000,
        max_test=5000,
    )

    # ── Base config ───────────────────────────────────────────────────────
    base = Config(
        num_classes=len(class_names),
        num_rounds=30,
        num_clients=10,
        local_epochs=2,
        local_lr=0.0005,
        server_lr=0.7,
        participation_ratio=0.8,
        batch_size=32,
        eval_batch_size=128,
        malicious_ratio=0.4,
        num_mc_samples=10,
        max_seq_length=256,
        seed=42,
        results_dir="results",
    )
    base.device = device

    # ── Experiment 1: Baseline (no defense) ────────────────────────────────
    cfg = copy.deepcopy(base)
    cfg.attack_type = "noise"
    cfg.experiment_name = "baseline_noise_no_defense"
    _stamp("Running baseline (no defense)...")
    baseline_f1s = run_experiment(cfg, train_ds, val_ds, test_ds, class_names)

    # ── Experiment 2: Defense ──────────────────────────────────────────────
    cfg2 = copy.deepcopy(base)
    cfg2.attack_type = "noise"
    cfg2.experiment_name = "defense_noise"
    controller = DefenseController(
        pos_sum_threshold=cfg2.defense_pos_sum_threshold,
        var_threshold=cfg2.defense_var_threshold,
    )
    _stamp("Running with defense...")
    defense_f1s = run_experiment(
        cfg2, train_ds, val_ds, test_ds, class_names,
        defense_controller=controller,
    )

    # ── Export defense history ────────────────────────────────────────────
    history = controller.get_history_df()
    export_defense_csv(history, cfg2.results_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30s} {'Final Macro-F1':>15s}")
    print("-" * 50)
    print(f"{'Baseline (no defense)':<30s} {baseline_f1s[-1]:>15.4f}")
    print(f"{'With DefenseController':<30s} {defense_f1s[-1]:>15.4f}")
    print("=" * 70)
    _stamp("All experiments complete.")


if __name__ == "__main__":
    main()
