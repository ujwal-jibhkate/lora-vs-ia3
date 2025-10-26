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

# src/metrics.py

# (Keep imports and other parts of the file the same)
...

def compute_metrics(eval_pred: EvalPrediction, task_name: str, tokenizer_name: str, token: str = None):
    """
    Computes task-specific metrics for evaluation.
    """
    predictions_or_tuple, labels = eval_pred # Rename to avoid confusion

    if task_name == "sst2":
        # Ensure predictions is the array if it's a tuple (less common here, but safe)
        predictions = predictions_or_tuple[0] if isinstance(predictions_or_tuple, tuple) else predictions_or_tuple
        preds = np.argmax(predictions, axis=1)

        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        results = {}
        results.update(acc_metric.compute(predictions=preds, references=labels))
        results.update(f1_metric.compute(predictions=preds, references=labels, average="binary"))

        return results

    elif task_name == "samsum":
        # --- FIX: Check if predictions is a tuple and get the first element ---
        if isinstance(predictions_or_tuple, tuple):
            predictions = predictions_or_tuple[0]
        else:
            predictions = predictions_or_tuple

        # Important: Ensure predictions is actually a numpy array now
        if not isinstance(predictions, np.ndarray):
             # This might happen if predict_with_generate=True in TrainingArgs (default for Seq2Seq)
             # And the output structure is unexpected. Let's log it.
             print(f"Warning: Unexpected predictions type: {type(predictions)}. Skipping ROUGE calculation.")
             return {} # Return empty if we can't process

        # Make a copy to avoid modifying the original tuple indirectly if predictions was a view
        predictions = np.copy(predictions)
        # --------------------------------------------------------------------

        rouge_metric = evaluate.load("rouge")
        tokenizer = _get_tokenizer(tokenizer_name, token=token)

        # Decode predictions (Now operates on the NumPy array)
        # Replace -100 (ignore index) with pad_token_id for decoding
        predictions[predictions == -100] = tokenizer.pad_token_id
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Decode labels
        # Make a copy of labels too, just to be safe
        labels = np.copy(labels)
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE requires newline-separated summaries
        decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds] # Added strip()
        decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels] # Added strip()

        result = rouge_metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {key: value * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    elif task_name == "dolly":
        return {} # Let the Trainer handle loss calculation

    # (Keep the temporary samsum block if you still have rouge_score issues locally)
    # elif task_name == "samsum":
    #     print("--- SKIPPING SAMSUM METRICS (LOCAL TEST) ---")
    #     return {}

    else:
        raise ValueError(f"No metrics defined for task: {task_name}")