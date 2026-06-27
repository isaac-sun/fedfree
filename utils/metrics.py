"""Evaluation metrics: loss and Macro-F1."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np


def evaluate_model(
    model,
    data_loader: DataLoader,
    device: str = "cpu",
) -> tuple[float, float]:
    """Evaluate model. Returns (loss, macro_f1)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attn_mask, labels = [b.to(device) for b in batch]
            logits = model(input_ids, attn_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, macro_f1
