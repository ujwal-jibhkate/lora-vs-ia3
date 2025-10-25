# tests/test_model_loader.py

import pytest
import os  # <-- Import os
from src.model_loader import load_base_model
from transformers import (
    DistilBertForSequenceClassification,
    T5ForConditionalGeneration,
    GemmaForCausalLM
)
from dotenv import load_dotenv  # <-- Import dotenv

# --- Load Environment Variables ---
load_dotenv() # This loads the .env file
HF_TOKEN = os.getenv("HF_TOKEN") # Reads the token
assert HF_TOKEN is not None, "HF_TOKEN not found in .env file. Please create .env and add HF_TOKEN=hf_..."
# ----------------------------------


# (TEST_CASES is unchanged)
TEST_CASES = [
  #  ("distilbert", "sst2", DistilBertForSequenceClassification),
   # ("t5", "samsum", T5ForConditionalGeneration),
   # ("gemma", "dolly", GemmaForCausalLM),
]

@pytest.mark.parametrize("model_name, task_name, expected_class", TEST_CASES)
def test_load_base_models_smoke_test(model_name, task_name, expected_class):
    # (Function description is unchanged)
    print(f"\nRunning smoke test for model: {model_name} with task: {task_name}")
    try:
        # --- Pass the token to the function ---
        model, tokenizer = load_base_model(
            model_name=model_name, 
            task_name=task_name,
            token=HF_TOKEN # <-- PASS TOKEN
        )
        
        # (All assertions are unchanged)
        assert model is not None, "Model should not be None"
        assert tokenizer is not None, "Tokenizer should not be None"
        
        assert isinstance(model, expected_class), \
            f"Model is wrong class. Expected {expected_class}, got {type(model)}"
            
        print(f"Successfully loaded model of type: {type(model)}")
        
        assert tokenizer.pad_token is not None, "Tokenizer must have a pad_token set"
        assert tokenizer.pad_token_id is not None, "Tokenizer must have a pad_token_id set"
        if model_name in ["t5", "gemma"]:
            assert model.config.pad_token_id == tokenizer.pad_token_id, "Model config pad_token_id must match tokenizer"
            
        print("Model and tokenizer configuration is correct.")

    except Exception as e:
        pytest.fail(f"Test failed for model {model_name} with exception: {e}")

def test_invalid_model_name():
    # (This test is unchanged)
    print("\nRunning test for invalid model name...")
    with pytest.raises(ValueError, match="Unknown model name"):
        load_base_model(model_name="not_a_real_model", task_name="sst2", token=HF_TOKEN)
    print("Correctly raised ValueError.")

def test_invalid_task_name():
    # (This test is unchanged)
    print("\nRunning test for invalid task name...")
    with pytest.raises(ValueError, match="Unknown task name"):
        load_base_model(model_name="distilbert", task_name="not_a_real_task", token=HF_TOKEN)
    print("Correctly raised ValueError.")