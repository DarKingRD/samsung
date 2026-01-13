"""
Module for loading and configuring a T5-based seq2seq model for the Russian language.

This module provides functions to load a tokenizer and a sequence-to-sequence model
using Hugging Face Transformers. The default model is `cointegrated/rut5-base`.

Key functions:
    - load_tokenizer: Loads a tokenizer for the specified model.
    - load_model: Loads the model and optionally moves it to a specific device (e.g., GPU).

Usage:
    >>> from model import load_tokenizer, load_model
    >>> tokenizer = load_tokenizer()
    >>> model = load_model(device="cuda")
"""

from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEFAULT_MODEL_NAME = "cointegrated/rut5-base"  # Model used by default


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> AutoTokenizer:
    """
    Loads and returns a tokenizer for the specified model.

    Args:
        model_name (str, optional): Name of the Hugging Face model or path to a local model.
            Defaults to "cointegrated/rut5-base".

    Returns:
        AutoTokenizer: Loaded tokenizer instance.
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
) -> AutoModelForSeq2SeqLM:
    """
    Loads a seq2seq model and optionally moves it to the specified device.

    Args:
        model_name (str, optional): Name of the Hugging Face model or path to a local model.
            Defaults to "cointegrated/rut5-base".
        device (str | None, optional): Device to move the model to ("cpu", "cuda", "cuda:0", etc.).
            If None, the model remains on the default device. Defaults to None.

    Returns:
        AutoModelForSeq2SeqLM: Loaded model instance.

    Note:
        The model is loaded in evaluation mode (`.eval()`) by default.
        Use `.train()` if fine-tuning is required.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    return model
