"""CSV export for defense analysis."""

import os
import pandas as pd


def export_defense_csv(history: list[dict], results_dir: str) -> str:
    """Export defense detection history to CSV.

    Returns the file path.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "defense_history.csv")

    df = pd.DataFrame(history)
    # Reorder columns for readability
    cols = ["round", "client_id", "is_attacker", "positive_sum",
            "variance", "suspected", "phase"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(path, index=False)
    print(f"[export] Defense history saved to {path}")
    return path
