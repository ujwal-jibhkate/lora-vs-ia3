# src/metrics.py

import evaluate
import numpy as np
from transformers import EvalPrediction, AutoTokenizer

# We will cache tokenizers, as they are slow to load.
# We will NOT cache metrics, as they load instantly.
TOKENIZERS_CACHE = {}

def _get_tokenizer(model_name_or_path, token=None):
    """Helper to load and cache tokenizers."""
    if model_name_or_path not in TOKENIZERS_CACHE:
        TOKENIZERS_CACHE[model_name_or_path] = AutoTokenizer.from_pretrained(
            model_name_or_path, token=token
        )
    return TOKENIZERS_CACHE[model_name_or_path]

def compute_metrics(eval_pred: EvalPrediction, task_name: str, tokenizer_name: str, token: str = None):
    """
    Computes task-specific metrics for evaluation.
    """
    predictions, labels = eval_pred

    if task_name == "sst2":
        preds = np.argmax(predictions, axis=1)
        
        # --- FIX: Load metrics on-the-fly ---
        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        
        results = {}
        results.update(acc_metric.compute(predictions=preds, references=labels))
        results.update(f1_metric.compute(predictions=preds, references=labels, average="binary")) 
        
        return results

    # We are temporarily disabling samsum due to local install issues.
    #
    elif task_name == "samsum":
        rouge_metric = evaluate.load("rouge") # <-- Load on the fly
        tokenizer = _get_tokenizer(tokenizer_name, token=token)
    
        predictions[predictions == -100] = tokenizer.pad_token_id
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
        decoded_labels = ["\n".join(label.split()) for label in decoded_labels]
    
        result = rouge_metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {key: value * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}
    
    elif task_name == "dolly":
        return {} # Let the Trainer handle loss calculation

    # Add this to handle the commented-out case
    elif task_name == "samsum":
        print("--- SKIPPING SAMSUM METRICS (LOCAL TEST) ---")
        return {} # Return empty dict for now

    else:
        raise ValueError(f"No metrics defined for task: {task_name}")