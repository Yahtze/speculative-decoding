"""
Simple interface to load models and tokenizers from Hugging Face.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_tokenizer(model_name: str, **kwargs) -> AutoTokenizer:
    """
    Load a tokenizer from Hugging Face.

    Args:
        model_name: Hugging Face model ID or local path.
        **kwargs: Additional kwargs passed to from_pretrained().

    Returns:
        AutoTokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str,
    device: str | None = None,
    dtype: torch.dtype | str | None = None,
    **kwargs,
) -> AutoModelForCausalLM:
    """
    Load a causal LM from Hugging Face.

    Args:
        model_name: Hugging Face model ID or local path.
        device: Target device ("cpu", "cuda", "mps"). Auto-detected if None.
        dtype: Model dtype (e.g. torch.float16, "auto"). Auto-detected if None.
        **kwargs: Additional kwargs passed to from_pretrained().

    Returns:
        AutoModelForCausalLM instance on the specified device.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    load_kwargs = {"device_map": device}

    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    load_kwargs.update(kwargs)

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model
