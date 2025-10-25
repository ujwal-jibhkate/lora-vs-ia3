# tests/test_data_loader.py

import pytest
import torch
import os  # <-- Import os
from datasets import Dataset
from src.data_loader import load_and_preprocess_dataset, DATASET_CONFIGS
from dotenv import load_dotenv  # <-- Import dotenv

# --- Load Environment Variables ---
load_dotenv() # This loads the .env file
HF_TOKEN = os.getenv("HF_TOKEN") # Reads the token
assert HF_TOKEN is not None, "HF_TOKEN not found in .env file. Please create .env and add HF_TOKEN=hf_..."
# ----------------------------------


@pytest.fixture(params=list(DATASET_CONFIGS.keys()))
def task_name(request):
    return request.param

def test_load_all_tasks_smoke_test(task_name):
    # (Function description is unchanged)
    print(f"\nRunning smoke test for task: {task_name}")
    
    # (split logic is unchanged)
    split = 'train[:10]'
    if task_name == "sst2":
        split = 'validation[:10]'
    if task_name == "samsum":
        split = 'validation[:10]'

    try:
        # --- Pass the token to the function ---
        dataset = load_and_preprocess_dataset(
            task_name=task_name, 
            split=split, 
            token=HF_TOKEN # <-- PASS TOKEN
        )
        
        # (All assertions are unchanged)
        assert dataset is not None, "Dataset should not be None"
        assert isinstance(dataset, Dataset), "Should return a Hugging Face Dataset"
        assert len(dataset) == 10, "Should have loaded exactly 10 examples"
        
        print(f"Loaded {len(dataset)} examples successfully.")
        
        required_cols = ["input_ids", "attention_mask", "labels"]
        for col in required_cols:
            assert col in dataset.features, f"Column '{col}' is missing"
            
        print(f"All required columns found: {list(dataset.features.keys())}")
        
        example = dataset[0]
        assert isinstance(example["input_ids"], torch.Tensor), "input_ids should be a torch.Tensor"
        assert isinstance(example["attention_mask"], torch.Tensor), "attention_mask should be a torch.Tensor"
        assert isinstance(example["labels"], torch.Tensor), "labels should be a torch.Tensor"
        
        if task_name == "sst2":
            assert example["input_ids"].shape == torch.Size([128])
            assert example["labels"].dim() == 0 
        elif task_name == "samsum":
            assert example["input_ids"].shape == torch.Size([512])
            assert example["labels"].shape == torch.Size([128])
        elif task_name == "dolly":
            assert example["input_ids"].shape == torch.Size([512])
            assert example["labels"].shape == torch.Size([512])
            
        print(f"Data types and shapes are correct.")
        
    except Exception as e:
        pytest.fail(f"Test failed for task {task_name} with exception: {e}")

def test_invalid_task_name():
    # (This test is unchanged)
    print("\nRunning test for invalid task name...")
    with pytest.raises(ValueError, match="Unknown task name"):
        load_and_preprocess_dataset(task_name="not_a_real_task")
    print("Correctly raised ValueError.")