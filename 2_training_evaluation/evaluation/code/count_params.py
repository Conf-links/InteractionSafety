"""Exact parameter counts for the architecture table (meta-device init: no weights, no GPU)."""
import json

import torch
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification

import common as C
from laya.common import build_model


def n(m):
    return sum(p.numel() for p in m.parameters())


out = {}
for key, name in [("deberta", "microsoft/deberta-v3-base"), ("promptguard", "meta-llama/Llama-Prompt-Guard-2-86M")]:
    with init_empty_weights():
        m = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(name, num_labels=2))
    out[key] = {"total": n(m), "trainable": n(m), "layers": m.config.num_hidden_layers, "hidden": m.config.hidden_size,
                "vocab": m.config.vocab_size}
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
for key, name in [("llamaguard3", "meta-llama/Llama-Guard-3-8B"), ("wildguard", "allenai/wildguard")]:
    cfg = AutoConfig.from_pretrained(name)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    total = n(m)
    pm = get_peft_model(m, lora)
    tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    out[key] = {"total": total, "trainable": tr, "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
                "vocab": cfg.vocab_size, "kv_heads": cfg.num_key_value_heads, "heads": cfg.num_attention_heads,
                "intermediate": cfg.intermediate_size}
laya_dir = C.Path.home() / "models" / "convaiinnovations-laya"
cfg = json.load(open(laya_dir / "rl_agent_config.json"))
with init_empty_weights():
    m = build_model(cfg, encoder_dir=str(laya_dir / "encoder"))
out["laya"] = {"total": n(m), "trainable": n(m), "encoder": n(m.encoder), "head": n(m) - n(m.encoder),
               "layers": m.encoder.config.num_hidden_layers, "hidden": m.encoder.config.hidden_size,
               "vocab": m.encoder.config.vocab_size, "head_layers": cfg["head_layers"]}
C.save_json(out, C.NEW_RESULTS / "metrics" / "model_params.json")
print(json.dumps(out, indent=1))
