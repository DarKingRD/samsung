from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEFAULT_MODEL_NAME = "cointegrated/rut5-base"


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    return model
