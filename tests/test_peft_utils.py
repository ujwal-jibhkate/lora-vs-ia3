# tests/test_peft_utils.py

import pytest
import os
from dotenv import load_dotenv
from src.model_loader import load_base_model
from src.peft_utils import apply_peft_adapter
from peft import PeftModel

# --- Load Environment Variables ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "HF_TOKEN not found in .env file."
print(f"--- Loaded HF_TOKEN starting with: {HF_TOKEN[:6]}... ---")
# ----------------------------------

# We will test all 3 models with both PEFT methods
MODEL_TASK_PAIRS = [
    ("distilbert", "sst2"),
    ("t5", "samsum"),
    # ("gemma", "dolly"), # <-- COMMENT THIS OUT IF YOU ARE LOW ON SPACE
]

PEFT_METHODS = ["lora", "ia3"]



@pytest.mark.parametrize("model_name, task_name", MODEL_TASK_PAIRS)
@pytest.mark.parametrize("peft_method", PEFT_METHODS)
def test_apply_peft_adapter(model_name, task_name, peft_method):
    """
    Tests that the PEFT adapter is applied correctly.
    It checks if the model is converted to a PeftModel and if the
    number of trainable parameters is a small fraction of the total.
    """
    print(f"\nTesting PEFT: {peft_method} on {model_name} for {task_name}")
    
    try:
        # 1. Load the base model
        base_model, _ = load_base_model(
            model_name=model_name,
            task_name=task_name,
            token=HF_TOKEN
        )
        total_params = sum(p.numel() for p in base_model.parameters())
        print(f"Base model loaded. Total params: {total_params}")

        # 2. Define a dummy PEFT config
        peft_config = {
            "peft_method": peft_method,
            "task_name": task_name,
            "r": 8, 
            "lora_alpha": 16,
        }

        # 3. Apply the adapter
        peft_model = apply_peft_adapter(base_model, peft_config)
        
        # --- Strict Validation Checks ---
        
        # FIX 1: This check MUST come first.
        # It ensures the model was successfully converted.
        assert isinstance(peft_model, PeftModel), \
            f"Model is not a PeftModel instance. Got {type(peft_model)} instead."
        
        # FIX 2: This is a more robust way to get trainable params
        # that doesn't rely on the helper function.
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        
        print(f"PEFT model created. Trainable params: {trainable_params}")
        
        assert trainable_params > 0, "No trainable parameters found!"
        
        # Check 3: Is the base model frozen?
        assert (trainable_params / total_params) < 0.05, \
            "Trainable params are > 5% of total. Base model is likely not frozen."
            
        print("PEFT adapter applied successfully and base model is frozen.")

    except Exception as e:
        if "No space left on device" in str(e):
            pytest.skip(f"Skipping {model_name} due to disk space limitations.")
        else:
            pytest.fail(f"Test failed for {peft_method} on {model_name} with exception: {e}")