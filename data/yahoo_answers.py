"""Yahoo Answers topic classification dataset via HuggingFace `datasets`.

10-class topic classification (Society, Science, Health, Education,
Computers, Sports, Business, Entertainment, Politics, Family).
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_yahoo_answers(
    model_name: str = "distilbert-base-uncased",
    max_seq_length: int = 256,
    val_ratio: float = 0.1,
    max_train: int = 20000,
    max_test: int = 5000,
    seed: int = 42,
):
    """Load Yahoo Answers, tokenize, return TensorDatasets + class names.

    Returns
    -------
    train_ds : TensorDataset   (input_ids, attention_mask, labels)
    val_ds   : TensorDataset
    test_ds  : TensorDataset
    class_names : list[str]    length = 10
    """
    print("[data] Loading Yahoo Answers from HuggingFace datasets...")

    raw = load_dataset("yahoo_answers_topics")

    # Subsample for FL feasibility
    train_raw = raw["train"].shuffle(seed=seed).select(range(min(len(raw["train"]), max_train * 2)))
    test_raw = raw["test"].shuffle(seed=seed).select(range(min(len(raw["test"]), max_test)))

    print(f"[data] Train: {len(train_raw)}, Test: {len(test_raw)}")
    class_names = raw["train"].features["topic"].names  # 10 topics
    print(f"[data] Classes: {len(class_names)} — {class_names}")

    # Tokenizer
    print(f"[data] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(examples):
        texts = [
            f"{title} {content}" if title and content else (title or content)
            for title, content in zip(
                examples["question_title"], examples["best_answer"]
            )
        ]
        tok = tokenizer(
            texts,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tok["labels"] = examples["topic"]
        return tok

    print("[data] Tokenizing (this may take a moment)...")
    train_tokenized = train_raw.map(_tokenize, batched=True, remove_columns=train_raw.column_names)
    test_tokenized = test_raw.map(_tokenize, batched=True, remove_columns=test_raw.column_names)

    # Convert to PyTorch tensors
    def _to_tensor(ds):
        return TensorDataset(
            torch.tensor(ds["input_ids"], dtype=torch.long),
            torch.tensor(ds["attention_mask"], dtype=torch.long),
            torch.tensor(ds["labels"], dtype=torch.long),
        )

    train_all = _to_tensor(train_tokenized)
    test_ds = _to_tensor(test_tokenized)

    # Split train → train + val
    train_indices, val_indices = train_test_split(
        range(len(train_all)),
        test_size=val_ratio,
        random_state=seed,
        stratify=[train_all[i][2].item() for i in range(len(train_all))],
    )
    train_ds = TensorDataset(
        train_all.tensors[0][train_indices],
        train_all.tensors[1][train_indices],
        train_all.tensors[2][train_indices],
    )
    val_ds = TensorDataset(
        train_all.tensors[0][val_indices],
        train_all.tensors[1][val_indices],
        train_all.tensors[2][val_indices],
    )

    print(f"[data] Ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds, class_names
