"""FL Server: manages global model, selects clients, evaluates."""

from __future__ import annotations

from collections import OrderedDict, deque

import numpy as np
from torch.utils.data import DataLoader

from utils.metrics import evaluate_model


class FLServer:
    """Federated learning server with global model management."""

    def __init__(self, model, val_dataset, test_dataset, config):
        self.model = model
        self.config = config

        from models.lora_model import get_lora_state_dict
        self.global_state_dict = get_lora_state_dict(model)
        self.global_history: deque = deque(maxlen=3)

        self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    def select_clients(self, num_clients: int, participation_ratio: float) -> list[int]:
        n = max(1, int(num_clients * participation_ratio))
        return sorted(np.random.choice(num_clients, n, replace=False).tolist())

    def get_global_state_dict(self) -> OrderedDict:
        from copy import deepcopy
        return deepcopy(self.global_state_dict)

    def update_global_model(self, new_state_dict: OrderedDict):
        self.global_history.append(
            OrderedDict({k: v.clone() for k, v in self.global_state_dict.items()})
        )
        self.global_state_dict = OrderedDict(
            {k: v.clone() for k, v in new_state_dict.items()}
        )

    def evaluate(self):
        from models.lora_model import load_lora_state_dict
        load_lora_state_dict(self.model, self.global_state_dict)
        self.model.to(self.config.device)
        return evaluate_model(self.model, self.test_loader, self.config.device)

    def evaluate_val(self):
        from models.lora_model import load_lora_state_dict
        load_lora_state_dict(self.model, self.global_state_dict)
        self.model.to(self.config.device)
        return evaluate_model(self.model, self.val_loader, self.config.device)
