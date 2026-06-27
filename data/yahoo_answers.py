"""Yahoo Answers topic classification — manual download via huggingface_hub.

Bypasses `datasets.load_dataset` entirely to avoid the
``huggingface-hub >= 0.26`` URI-parsing regression on Colab.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset, Subset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download

# ── Cache for test data (downloaded once per session) ────────────────────────
_CACHE: dict[str, tuple] = {}


def load_yahoo_answers(
    model_name: str = "distilbert-base-uncased",
    max_seq_length: int = 256,
    val_ratio: float = 0.1,
    max_train: int = 20000,
    max_test: int = 5000,
    seed: int = 42,
):
    """Load Yahoo Answers, tokenize, return TensorDatasets + class names.

    Downloads the raw CSV from HF Hub directly (no ``datasets.load_dataset``
    codepath) to avoid the ``huggingface-hub >= 0.26`` URI-parsing regression.
    """
    import pandas as pd

    cache_key = (max_train, max_test, seed)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    print("[data] Downloading Yahoo Answers from HuggingFace Hub...")

    # ── Download CSV files directly ───────────────────────────────────────
    train_path = hf_hub_download(
        repo_id="yahoo_answers_topics",
        filename="data/train.csv",
        repo_type="dataset",
    )
    test_path = hf_hub_download(
        repo_id="yahoo_answers_topics",
        filename="data/test.csv",
        repo_type="dataset",
    )

    # ── Load with pandas ──────────────────────────────────────────────────
    class_names = [
        "Society & Culture", "Science & Mathematics", "Health",
        "Education & Reference", "Computers & Internet", "Sports",
        "Business & Finance", "Entertainment & Music",
        "Family & Relationships", "Politics & Government",
    ]

    df_train = pd.read_csv(train_path, header=None, names=["label", "title", "content", "answer"])
    df_test = pd.read_csv(test_path, header=None, names=["label", "title", "content", "answer"])

    # Labels are 1-indexed in the CSV; shift to 0-indexed
    df_train["label"] = df_train["label"].astype(int) - 1
    df_test["label"] = df_test["label"].astype(int) - 1

    print(f"[data] Train: {len(df_train)}, Test: {len(df_test)}")
    # ── Build texts ───────────────────────────────────────────────────────
    def _build_text(row):
        t = str(row["title"]) if pd.notna(row["title"]) else ""
        c = str(row["content"]) if pd.notna(row["content"]) else ""
        return f"{t} {c}".strip() or " "

    train_texts = [_build_text(r) for _, r in df_train.iterrows()]
    test_texts = [_build_text(r) for _, r in df_test.iterrows()]
    train_labels = df_train["label"].tolist()
    test_labels = df_test["label"].tolist()

    print(f"[data] Classes: {len(class_names)} — {class_names}")

    # ── Tokenize ──────────────────────────────────────────────────────────
    print(f"[data] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tok(texts):
        enc = tokenizer(
            texts, max_length=max_seq_length,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    print("[data] Tokenizing (this may take a moment)...")
    train_ids, train_mask = _tok(train_texts)
    test_ids, test_mask = _tok(test_texts)

    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)

    # ── Build TensorDatasets ──────────────────────────────────────────────
    train_all = TensorDataset(train_ids, train_mask, train_labels_t)
    test_ds = TensorDataset(test_ids, test_mask, test_labels_t)

    # Split train → train + val
    n = len(train_all)
    indices = list(range(n))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, random_state=seed,
        stratify=[train_all[i][2].item() for i in indices],
    )
    train_ds = Subset(train_all, train_idx)
    val_ds = Subset(train_all, val_idx)

    result = (train_ds, val_ds, test_ds, class_names)
    _CACHE[cache_key] = result
    print(f"[data] Ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return result
