# src/model_loader.py

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch

# (TASK_TO_MODEL_CLASS and MODEL_NAME_MAP are unchanged)
TASK_TO_MODEL_CLASS = {
    "sst2": AutoModelForSequenceClassification,
    "samsum": AutoModelForSeq2SeqLM,
    "dolly": AutoModelForCausalLM,
}

MODEL_NAME_MAP = {
    "distilbert": "distilbert-base-uncased",
    "t5": "t5-small",
    #"gemma": "google/gemma-2b",
    "pythia": "EleutherAI/pythia-2.8b"
}


def load_base_model(model_name: str, task_name: str, token: str = None): # <-- FIX: Added token
    """
    Loads a base model and its tokenizer, configured for 8-bit loading.

    Args:
        model_name (str): One of 'distilbert', 't5', or 'gemma'.
        task_name (str): One of 'sst2', 'samsum', or 'dolly'.
        token (str, optional): Hugging Face token for gated models.

    Returns:
        A tuple of (model, tokenizer)
    """
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(f"Unknown model name: {model_name}. Must be {list(MODEL_NAME_MAP.keys())}")
    
    if task_name not in TASK_TO_MODEL_CLASS:
        raise ValueError(f"Unknown task name: {task_name}. Must be {list(TASK_TO_MODEL_CLASS.keys())}")

    hf_model_name = MODEL_NAME_MAP[model_name]
    model_class = TASK_TO_MODEL_CLASS[task_name]
    
    # (bnb_config is unchanged)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name, 
        token=token  # <-- FIX: Pass token
    )
    
    # (pad token logic is unchanged)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Load Model
    try:
        model = model_class.from_pretrained(
            hf_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=token, 
            trust_remote_code=True
        )
    except Exception as e:
        # (Fallback logic is unchanged)
        print(f"8-bit loading failed (this is expected on CPU-only machines): {e}")
        print("Falling back to standard precision (float32) loading.")
        model = model_class.from_pretrained(
            hf_model_name,
            token=token,  # <-- FIX: Pass token here too
            trust_remote_code=True
        )
        
    # (pad_token_id config is unchanged)
    if model_name in ["t5", "gemma"]:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer