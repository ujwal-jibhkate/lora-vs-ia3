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
    predictions_or_tuple, labels = eval_pred

    if task_name == "sst2":
        # (sst2 block remains the same)
        predictions = predictions_or_tuple[0] if isinstance(predictions_or_tuple, tuple) else predictions_or_tuple
        preds = np.argmax(predictions, axis=1)
        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        results = {}
        results.update(acc_metric.compute(predictions=preds, references=labels))
        results.update(f1_metric.compute(predictions=preds, references=labels, average="binary"))
        return results

    elif task_name == "samsum":
        if isinstance(predictions_or_tuple, tuple):
            predictions = predictions_or_tuple[0]
        else:
            predictions = predictions_or_tuple

        if not isinstance(predictions, np.ndarray):
             print(f"Warning: Unexpected predictions type: {type(predictions)}. Skipping ROUGE calculation.")
             return {}

        # Ensure integer type first
        predictions = np.copy(predictions).astype(np.int64)
        labels = np.copy(labels).astype(np.int64)

        rouge_metric = evaluate.load("rouge")
        tokenizer = _get_tokenizer(tokenizer_name, token=token)

        # --- FIX: Convert NumPy arrays to Python lists of lists ---
        # Replace -100 with pad_token_id *before* converting to list
        predictions[predictions == -100] = tokenizer.pad_token_id
        labels[labels == -100] = tokenizer.pad_token_id

        # Convert to list of lists
        pred_ids_list = predictions.tolist()
        label_ids_list = labels.tolist()
        # --------------------------------------------------------

        # Decode using the lists
        decoded_preds = tokenizer.batch_decode(pred_ids_list, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_ids_list, skip_special_tokens=True)

        # ROUGE post-processing
        decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
        decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]

        result = rouge_metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Make sure result is not None (can happen with empty preds/refs)
        if result is None:
            print("Warning: ROUGE computation returned None. Returning empty metrics.")
            return {}
        result = {key: value * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    elif task_name == "dolly":
        # (dolly block remains the same)
        return {}

    # (Keep the temporary samsum block if you still have rouge_score issues locally)
    # elif task_name == "samsum":
    #     print("--- SKIPPING SAMSUM METRICS (LOCAL TEST) ---")
    #     return {}

    else:
        raise ValueError(f"No metrics defined for task: {task_name}")
