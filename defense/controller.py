"""
Defense Controller: two-phase free-rider detection via per-class Shapley.

Phase 1 — Positive-Sum Filter:
    positive_class_sv_sum = Σ_c max(SV_{i,c}, 0)
    If sum ≈ 0 → client contributes nothing → flag as suspected.

Phase 2 — Variance Fingerprinting:
    class_sv_variance = Var_c(SV_{i,c})
    Honest nodes: high variance (specialize in certain classes).
    Disguised free-riders: low variance (flat / near-zero per-class SV).
"""

from __future__ import annotations

import numpy as np


def compute_class_sv_metrics(
    per_class_sv: dict[int, np.ndarray],
    eps: float = 1e-10,
) -> dict[int, dict[str, float]]:
    """Compute per-class SV metrics for each client.

    Returns dict: client_id -> {
        'positive_class_sv_sum':  float,
        'class_sv_variance':      float,
        'dominant_class':         int,
    }
    """
    metrics = {}
    for cid, sv_arr in per_class_sv.items():
        positive = np.maximum(sv_arr, 0.0)
        a_i = float(np.sum(positive))
        v_i = float(np.var(sv_arr))
        c_i = int(np.argmax(positive))

        metrics[cid] = {
            "positive_class_sv_sum": a_i,
            "class_sv_variance": v_i,
            "dominant_class": c_i,
        }
    return metrics


class DefenseController:
    """Two-phase free-rider detector based on per-class Shapley values.

    Parameters
    ----------
    pos_sum_threshold : float
        Phase 1: clients with positive_sum < threshold are immediately
        suspected as trivial free-riders (noise/zero).
    var_threshold : float
        Phase 2: among surviving clients, those with variance < threshold
        are classified as disguised free-riders.
    """

    def __init__(
        self,
        pos_sum_threshold: float = 0.01,
        var_threshold: float = 0.001,
    ):
        self.pos_sum_threshold = pos_sum_threshold
        self.var_threshold = var_threshold

        # Tracking for analysis
        self.round_history: list[dict] = []

    def detect(
        self,
        per_class_sv: dict[int, np.ndarray],
        client_ids: list[int],
        attacker_ids: set[int],
        round_num: int,
    ) -> dict[int, dict]:
        """Run two-phase detection.

        Returns
        -------
        results : dict  client_id -> {
            'positive_sum': float,
            'variance':     float,
            'suspected':    bool,     # flagged in either phase
            'phase':        str,      # 'none', 'phase1_pos_sum', 'phase2_variance'
            'is_attacker':  bool,     # ground truth
        }
        """
        sv_metrics = compute_class_sv_metrics(per_class_sv)
        results = {}

        for cid in client_ids:
            m = sv_metrics.get(cid, {"positive_class_sv_sum": 0.0, "class_sv_variance": 0.0})
            pos_sum = m["positive_class_sv_sum"]
            var_val = m["class_sv_variance"]
            is_attacker = cid in attacker_ids

            phase = "none"
            suspected = False

            # Phase 1: Positive-Sum Filter
            if pos_sum < self.pos_sum_threshold:
                phase = "phase1_pos_sum"
                suspected = True
            # Phase 2: Variance Fingerprinting
            elif var_val < self.var_threshold:
                phase = "phase2_variance"
                suspected = True

            results[cid] = {
                "positive_sum": pos_sum,
                "variance": var_val,
                "suspected": suspected,
                "phase": phase,
                "is_attacker": is_attacker,
                "round": round_num,
                "client_id": cid,
            }

        self.round_history.extend(results.values())
        return results

    def get_history_df(self):
        """Return round history as a list of dicts (for CSV export)."""
        return list(self.round_history)
