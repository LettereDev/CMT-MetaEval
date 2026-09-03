# src/models.py
import gc
import torch
from .config import MODELS

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


def load_model(model_id):
    # Unload any previously loaded model to free GPU memory
    model_to_unload = [name for name in MODELS.values() if name != model_id]
    if model_to_unload in globals():
        unload_model()

    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )

    return tokenizer, model


def unload_model():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Model unloaded.")