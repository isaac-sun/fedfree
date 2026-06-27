"""Yahoo Answers topic classification — Parquet download via huggingface_hub.

Bypasses ``datasets.load_dataset`` entirely to avoid the
``huggingface-hub >= 0.26`` URI-parsing regression on Colab.
Downloads raw Parquet files directly and reads with pandas.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset, Subset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, list_repo_files

# ── Cache for test data (downloaded once per session) ────────────────────────
_CACHE: dict[tuple, tuple] = {}

CLASS_NAMES = [
    "Society & Culture", "Science & Mathematics", "Health",
    "Education & Reference", "Computers & Internet", "Sports",
    "Business & Finance", "Entertainment & Music",
    "Family & Relationships", "Politics & Government",
]

REPO_ID = "yahoo_answers_topics"
DATA_PREFIX = "yahoo_answers_topics"


def load_yahoo_answers(
    model_name: str = "distilbert-base-uncased",
    max_seq_length: int = 256,
    val_ratio: float = 0.1,
    max_train: int = 20000,
    max_test: int = 5000,
    seed: int = 42,
):
    """Load Yahoo Answers, tokenize, return TensorDatasets + class names.

    Downloads Parquet files from HF Hub directly — no ``datasets.load_dataset``
    codepath — then subsamples, tokenizes, and splits.
    """
    import pandas as pd

    cache_key = (max_train, max_test, seed)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    print("[data] Downloading Yahoo Answers from HuggingFace Hub...")

    # ── List repo files to discover Parquet shards ────────────────────────
    repo_files = list_repo_files(REPO_ID, repo_type="dataset")

    def _load_split(split: str) -> pd.DataFrame:
        """Download and concatenate all Parquet shards for a split."""
        prefix = f"{DATA_PREFIX}/{split}-"
        shards = sorted(
            f for f in repo_files
            if f.startswith(prefix) and f.endswith(".parquet")
        )
        if not shards:
            raise FileNotFoundError(
                f"No Parquet files found for split '{split}' in {REPO_ID}"
            )
        dfs = []
        for filename in shards:
            path = hf_hub_download(
                repo_id=REPO_ID, filename=filename, repo_type="dataset",
            )
            dfs.append(pd.read_parquet(path))
        return pd.concat(dfs, ignore_index=True)

    df_train = _load_split("train")
    df_test = _load_split("test")

    print(f"[data] Full dataset — Train: {len(df_train)}, Test: {len(df_test)}")

    # ── Subsample for FL feasibility ──────────────────────────────────────
    if max_train and len(df_train) > max_train:
        df_train = df_train.sample(n=max_train, random_state=seed)
    if max_test and len(df_test) > max_test:
        df_test = df_test.sample(n=max_test, random_state=seed)

    print(f"[data] After sampling — Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"[data] Classes: {len(CLASS_NAMES)} — {CLASS_NAMES}")

    # ── Build texts from question_title + best_answer ─────────────────────
    def _build_text(row) -> str:
        title = str(row["question_title"]) if pd.notna(row["question_title"]) else ""
        answer = str(row["best_answer"]) if pd.notna(row["best_answer"]) else ""
        return f"{title} {answer}".strip() or " "

    train_texts = [_build_text(r) for _, r in df_train.iterrows()]
    test_texts = [_build_text(r) for _, r in df_test.iterrows()]
    train_labels = df_train["topic"].astype(int).tolist()
    test_labels = df_test["topic"].astype(int).tolist()

    # ── Tokenize ──────────────────────────────────────────────────────────
    print(f"[data] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tok(texts: list[str]):
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

    result = (train_ds, val_ds, test_ds, CLASS_NAMES)
    _CACHE[cache_key] = result
    print(f"[data] Ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return result
