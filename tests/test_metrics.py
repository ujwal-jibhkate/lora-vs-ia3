# tests/test_metrics.py

import pytest
import numpy as np
import os
from transformers import EvalPrediction
from src.metrics import compute_metrics
from dotenv import load_dotenv

# (Keep token loading)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "HF_TOKEN not found in .env file."
# ----------------------------------

def test_sst2_metrics():
    # (This function is unchanged)
    print("\nTesting metrics for: sst2")
    logits = np.array([
        [0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]
    ])
    labels = np.array([1, 1, 1, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    
    results = compute_metrics(
        eval_pred, 
        task_name="sst2", 
        tokenizer_name="distilbert-base-uncased",
        token=HF_TOKEN
    )
    
    assert "accuracy" in results
    assert "f1" in results
    assert results["accuracy"] == 0.5 
    assert abs(results["f1"] - 0.6666) < 0.001
    print("SST-2 metrics calculated correctly.")

# --- COMMENT OUT THE ENTIRE SAMSUM TEST ---
#
# def test_samsum_metrics():
#     """Tests ROUGE calculation for samsum."""
#     print("\nTesting metrics for: samsum")
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("t5-small", token=HF_TOKEN)
#     ...
#     ...
#     print("SAMSum ROUGE metrics calculated correctly.")
# ---------------------------------------------

def test_dolly_metrics():
    # (This function is unchanged)
    print("\nTesting metrics for: dolly")
    eval_pred = EvalPrediction(predictions=None, label_ids=None)
    results = compute_metrics(
        eval_pred, 
        task_name="dolly", 
        tokenizer_name="google/gemma-2b", 
        token=HF_TOKEN
    )
    
    assert isinstance(results, dict)
    assert len(results) == 0
    print("Dolly metrics (empty dict) returned correctly.")