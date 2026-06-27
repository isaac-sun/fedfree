"""
Configuration for FL free-rider defense experiments.

All hyperparameters are centralized here.  Modify defaults or override
at construction time.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────
    model_name: str = "distilbert-base-uncased"
    num_classes: int = 10                     # Yahoo Answers
    max_seq_length: int = 512
    val_ratio: float = 0.1

    # ── Subsampling (Yahoo Answers has 1.4M train samples) ────────────────
    max_train_samples: int = 20000            # subsample for FL feasibility
    max_test_samples: int = 5000

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_lin", "k_lin", "v_lin", "out_lin")

    # ── Federated Learning ────────────────────────────────────────────────
    num_clients: int = 10
    num_rounds: int = 30
    local_epochs: int = 2
    local_lr: float = 0.0005
    server_lr: float = 0.7
    participation_ratio: float = 0.8
    batch_size: int = 32
    label_smoothing: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # ── Attack ────────────────────────────────────────────────────────────
    attack_type: str = "none"    # "none", "dfr", "sdfr", "afr"
    malicious_ratio: float = 0.4 # fraction of clients that attack

    # DFR: σ · t^(-γ) · N(0,I)  — Fraboni et al.
    dfr_sigma: float = 0.5
    dfr_gamma: float = 1.0
    dfr_estimate_sigma: bool = True  # auto-estimate σ from first global delta

    # AFR: SDFR + calibrated sparse noise  — Zhu et al.
    afr_e_cos_beta_override: float | None = None  # if set, use fixed E[cos β]
    afr_noisy_frac: float = 0.1                  # fraction of params to perturb
    afr_base_norm_ema_alpha: float = 0.3          # EMA smooth factor for base norm

    # ── Shapley (defense) ─────────────────────────────────────────────────
    num_mc_samples: int = 10                  # Monte Carlo permutations
    eval_batch_size: int = 128

    # ── Defense thresholds ────────────────────────────────────────────────
    defense_pos_sum_threshold: float = 0.01   # positive-sum < this → suspected
    defense_var_threshold: float = 0.001      # variance < this → confirmed

    # ── General ───────────────────────────────────────────────────────────
    seed: int = 42
    device: str = field(default_factory=_detect_device)
    results_dir: str = "results"
    experiment_name: str = "default"
