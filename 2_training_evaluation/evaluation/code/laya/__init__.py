"""Minimal `laya` package rebuilt from the official Hugging Face repo (convaiinnovations/laya).

`laya.common` is the repo's `rl_common.py` copied verbatim; `laya.load` mirrors `rl_agent_api.RLAgent`
(tokenizer + DecisionModel with weights) so FineTuneLaya*.py style code runs unchanged.
"""
import json
import os

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from . import common


class Agent:
    def __init__(self, model_dir, device="cpu"):
        with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
            self.cfg = json.load(f)
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.model = common.build_model(self.cfg, encoder_dir=os.path.join(model_dir, "encoder"))
        self.model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
        self.model.encoder.config.reference_compile = False
        self.model.to(torch.device(device))


def load(model_dir, device="cpu"):
    return Agent(model_dir, device)
