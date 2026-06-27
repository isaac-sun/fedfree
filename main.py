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
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config
from utils.seed import set_seed
from utils.metrics import evaluate_model
from data.yahoo_answers import load_yahoo_answers
from models.lora_model import create_model, get_lora_state_dict, load_lora_state_dict
from fl.client import FLClient
from fl.server import FLServer
from fl.fedavg import fedavg_aggregate
from attacks import dfr_attack, sdfr_attack, afr_attack, estimate_dfr_sigma, AFRState
from defense.shapley import (
    estimate_round_shapley_per_class,
    per_class_to_overall,
    _class_weights_from_loader,
)


def _stamp(msg: str):
    print(f"[fedfree] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Single Experiment
# ═══════════════════════════════════════════════════════════════════════════════



def _apply_attack(
    attack_type: str,
    global_sd,
    global_history: list,
    config,
    round_num: int = 1,
    dfr_sigma_est: float | None = None,
    afr_state: AFRState | None = None,
    val_loss_t: float | None = None,
):
    """Generate a free-rider update. Returns (update, meta_dict)."""
    if attack_type == "dfr":
        sigma = config.dfr_sigma
        if config.dfr_estimate_sigma and dfr_sigma_est is not None:
            sigma = dfr_sigma_est
        update = dfr_attack(global_sd, sigma=sigma,
                            round_num=round_num, gamma=config.dfr_gamma)
        return update, {}
    elif attack_type == "sdfr":
        update = sdfr_attack(global_sd, global_history)
        return update, {}
    elif attack_type == "afr":
        e_cos_beta = 0.0
        if config.afr_e_cos_beta_override is not None:
            e_cos_beta = config.afr_e_cos_beta_override
        elif afr_state is not None and val_loss_t is not None:
            e_cos_beta = afr_state.get_e_cos_beta(val_loss_t)
        mean_base_norm = None
        if afr_state is not None:
            mean_base_norm = afr_state.get_mean_base_norm()
        update, base_norm = afr_attack(
            global_sd, global_history,
            n_total=config.num_clients,
            e_cos_beta=e_cos_beta,
            mean_base_norm=mean_base_norm,
            noisy_frac=config.afr_noisy_frac,
        )
        return update, {"afr_base_norm": base_norm}
    # fallback: honest training → shouldn't be reached for attackers
    return global_sd, {}
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


    # ── Attack state tracking ──────────────────────────────────────────────
    dfr_sigma_est = None
    afr_state = AFRState(ema_alpha=config.afr_base_norm_ema_alpha) if config.attack_type == "afr" else None
    # ── Tracking ───────────────────────────────────────────────────────────
    test_f1s = []
    summary_rows = []

    # ── FL Rounds ──────────────────────────────────────────────────────────
    for round_t in tqdm(range(config.num_rounds), desc=f"{tag} {config.attack_type}"):
        selected = server.select_clients(config.num_clients, config.participation_ratio)
        global_sd = server.get_global_state_dict()
        global_history = list(server.global_history)

        # ── Pre-attack: DFR sigma estimation ─────────────────────────────
        if config.attack_type == "dfr" and dfr_sigma_est is None and config.dfr_estimate_sigma:
            est = estimate_dfr_sigma(global_sd, global_history)
            if est is not None:
                dfr_sigma_est = est
                _stamp(f"DFR sigma auto-estimated: {dfr_sigma_est:.6f}")

        # ── Pre-attack: validation loss for AFR state ────────────────────
        val_loss_t = None
        if config.attack_type == "afr":
            val_loss_t, _ = server.evaluate_val()
            if afr_state is not None and afr_state.val_loss_init is None:
                afr_state.val_loss_init = val_loss_t
                _stamp(f"AFR val_loss_init set: {val_loss_t:.4f}")

        # Collect client updates
        updates = {}
        afr_base_norms_this_round = []
        for cid in selected:
            if cid in attacker_ids:
                update, meta = _apply_attack(
                    config.attack_type, global_sd, global_history, config,
                    round_num=round_t + 1,
                    dfr_sigma_est=dfr_sigma_est,
                    afr_state=afr_state,
                    val_loss_t=val_loss_t,
                )
                updates[cid] = update
                if "afr_base_norm" in meta:
                    afr_base_norms_this_round.append(meta["afr_base_norm"])
            else:
                updates[cid] = clients[cid].train(global_sd)

        # ── Post-attack: update AFR state ────────────────────────────────
        if afr_state is not None and val_loss_t is not None and afr_base_norms_this_round:
            avg_base_norm = float(np.mean(afr_base_norms_this_round))
            afr_state.update(val_loss_t, avg_base_norm)

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
        max_seq_length=256,
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

    # ── Experiment grid (same as 20NEWS-FL) ────────────────────────────────
    experiments = [
        ("baseline_no_attack", "none"),
        ("attack_dfr",         "dfr"),
        ("attack_sdfr",        "sdfr"),
        ("attack_afr",         "afr"),
    ]

    all_f1s: dict[str, list[float]] = {}

    for exp_name, attack_type in experiments:
        cfg = copy.deepcopy(base)
        cfg.experiment_name = exp_name
        cfg.attack_type = attack_type

        _stamp(f"Running: {exp_name}")
        f1s = run_experiment(cfg, train_ds, val_ds, test_ds, class_names)
        all_f1s[exp_name] = f1s
        _stamp(f"  Done: {exp_name} — final F1={f1s[-1]:.4f}")

    # ── Save F1 curves ─────────────────────────────────────────────────────
    os.makedirs(base.results_dir, exist_ok=True)
    curves_path = os.path.join(base.results_dir, "f1_curves.csv")
    curves_df = pd.DataFrame({
        exp_name: f1s for exp_name, f1s in all_f1s.items()
    })
    curves_df.to_csv(curves_path, index_label="round")
    _stamp(f"F1 curves saved to {curves_path}")

    # ── Summary table ──────────────────────────────────────────────────────
    baseline_final = all_f1s["baseline_no_attack"][-1]
    print("\n" + "=" * 75)
    print("EXPERIMENT SUMMARY")
    print("=" * 75)
    print(f"{'Experiment':<25s} {'Final F1':>10s} {'Δ vs Baseline':>15s}")
    print("-" * 55)
    for exp_name, attack_type in experiments:
        final = all_f1s[exp_name][-1]
        delta = final - baseline_final
        print(f"  {exp_name:<23s} {final:>10.4f} {delta:>+15.4f}")
    print("=" * 75)

    _stamp("All experiments complete.")

if __name__ == "__main__":
    print("[main.py] entering main()...", flush=True)
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("\n[main.py] CRASHED — see traceback above", flush=True)
        import sys as _sys
        _sys.exit(1)
