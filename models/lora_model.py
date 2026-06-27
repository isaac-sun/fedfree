"""DistilBERT + LoRA classifier via PEFT library."""

from __future__ import annotations

import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import AutoModel

# ── Workaround: peft >= 0.15 requires torchao >= 0.16 but we use standard
# LoRA. Must patch is_torchao_available BEFORE importing peft, because peft's
# submodules may cache a local reference during import.
import sys as _sys
import peft.import_utils as _peft_iu
_peft_orig = _peft_iu.is_torchao_available


def _patched_is_torchao_available():
    try:
        return _peft_orig()
    except ImportError:
        return False


_peft_iu.is_torchao_available = _patched_is_torchao_available

from peft import LoraConfig, get_peft_model, TaskType



def create_model(
    model_name: str = "distilbert-base-uncased",
    num_classes: int = 10,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_target_modules: tuple = ("q_lin", "k_lin", "v_lin", "out_lin"),
):
    """Build a DistilBERT + LoRA classifier via PEFT.

    Returns
    -------
    model : PeftModel   (wraps DistilBERT with LoRA adapters + classifier head)
    """
    # Ensure torchao module sees our patched version
    if "peft.tuners.lora.torchao" in _sys.modules:
        _sys.modules["peft.tuners.lora.torchao"].is_torchao_available = _patched_is_torchao_available

    backbone = AutoModel.from_pretrained(model_name)

    # Wrap backbone in a classification model
    class DistilBERTClassifier(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.distilbert = backbone
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(backbone.config.dim, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            pooled = self.dropout(pooled)
            return self.classifier(pooled)

    model = DistilBERTClassifier(backbone, num_classes)

    # Apply LoRA via PEFT
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules),
    )
    model = get_peft_model(model, peft_config)

    # Ensure classifier head is trainable
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    return model


def get_lora_state_dict(model) -> OrderedDict:
    """Extract only trainable parameters (LoRA + head)."""
    return OrderedDict(
        {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    )


def load_lora_state_dict(model, state_dict: OrderedDict):
    """Load trainable parameters into the model."""
    model_sd = model.state_dict()
    for key, val in state_dict.items():
        if key in model_sd:
            model_sd[key].copy_(val.to(model_sd[key].dtype))
    model.load_state_dict(model_sd, strict=False)
