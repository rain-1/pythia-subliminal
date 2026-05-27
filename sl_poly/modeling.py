from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelLoadConfig
from .utils import device_or_cpu, torch_dtype


def load_tokenizer(model_name: str, trust_remote_code: bool = False):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: ModelLoadConfig):
    device = device_or_cpu(cfg.device)
    kwargs = {"trust_remote_code": cfg.trust_remote_code}
    if cfg.revision:
        kwargs["revision"] = cfg.revision
    if device != "cpu":
        kwargs["torch_dtype"] = torch_dtype(cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    model.to(device)
    model.eval()
    return model
