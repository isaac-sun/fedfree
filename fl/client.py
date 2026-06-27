"""Federated Learning client with local LoRA training."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.lora_model import create_model, get_lora_state_dict, load_lora_state_dict


class FLClient:
    """Local client: loads global state, trains locally, returns delta."""

    def __init__(self, client_id: int, dataset, config):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

        loader_kwargs = dict(batch_size=config.batch_size, shuffle=True, drop_last=False)
        if config.device == "cuda":
            loader_kwargs.update(num_workers=2, pin_memory=True)
        self.data_loader = DataLoader(dataset, **loader_kwargs)

    def train(self, global_state_dict: OrderedDict) -> OrderedDict:
        """Local training: returns delta = w_local - w_global (trainable params only)."""
        model = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
        )
        load_lora_state_dict(model, global_state_dict)
        model.to(self.config.device)
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.local_lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        total_steps = self.config.local_epochs * len(self.data_loader)
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        global_step = 0
        for _ in range(self.config.local_epochs):
            for batch in self.data_loader:
                input_ids, attn_mask, labels = [b.to(self.config.device) for b in batch]

                # Linear warmup
                if global_step < warmup_steps:
                    lr_scale = (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = self.config.local_lr * lr_scale

                optimizer.zero_grad()
                logits = model(input_ids, attn_mask)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, self.config.max_grad_norm)
                optimizer.step()
                global_step += 1

        # Compute delta: only trainable params
        local_sd = get_lora_state_dict(model)
        update = OrderedDict()
        for key in local_sd:
            update[key] = local_sd[key] - global_state_dict[key].to(
                local_sd[key].device, dtype=local_sd[key].dtype
            )
        return update
