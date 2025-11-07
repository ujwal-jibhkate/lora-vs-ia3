# tests/test_trainer.py

import pytest
from pytest import fail, skip 
import os
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv() 
HF_TOKEN = os.getenv("HF_TOKEN") 
assert HF_TOKEN is not None, "HF_TOKEN not found in .env file. Please create .env and add HF_TOKEN=hf_..."
print(f"--- Loaded HF_TOKEN starting with: {HF_TOKEN[:6]}... ---") # Debug print
# ----------------------------------

DUMMY_CONFIG = {
    "task_name": "sst2",
    "model_name": "distilbert",
    "peft_method": "lora",
    "run_name": "test_run_lora_r8",
    "r": 8,
    "lora_alpha": 16,
    "num_epochs": 1,
    "batch_size": 2,
    "wandb_project": "test_project" 
}

@pytest.fixture
def mock_dependencies(mocker):
    """
    This fixture mocks (replaces) all our slow, heavy functions.
    It uses local imports to avoid circular dependency issues during collection.
    """
    # Import locally inside the fixture
    import src.trainer

    # Mock wandb
    mock_wandb_init = mocker.patch.object(src.trainer, "wandb", MagicMock())
    # Configure the mock returned by wandb.init()
    mock_run = MagicMock(summary={})
    mock_wandb_init.init.return_value = mock_run

    # Mock data loader
    # Make it return a mock dataset with a split method for the Dolly validation case
    mock_dataset = MagicMock()
    mock_dataset.train_test_split.return_value = {'train': 'fake_train_split', 'test': 'fake_eval_split'}
    mock_load_data = mocker.patch.object(
        src.trainer,
        "load_and_preprocess_dataset",
        return_value=mock_dataset # Return the mock dataset object
    )

    # Mock model loader
    mock_tokenizer = MagicMock()
    mock_base_model = MagicMock()
    mock_load_model = mocker.patch.object(
        src.trainer,
        "load_base_model",
        return_value=(mock_base_model, mock_tokenizer)
    )

    # Mock peft utils
    mock_peft_model = MagicMock()
    mock_peft_model.get_num_trainable_parameters.return_value = 1000 
    # Mock parameters() to avoid ZeroDivisionError
    mock_param_1 = MagicMock()
    mock_param_1.numel.return_value = 100_000_000
    mock_param_1.requires_grad = False # Base model param
    mock_param_2 = MagicMock()
    mock_param_2.numel.return_value = 1_000     # Adapter param
    mock_param_2.requires_grad = True
    mock_peft_model.parameters.return_value = [mock_param_1, mock_param_2] # List of mock params
    mock_apply_peft = mocker.patch.object(
        src.trainer,
        "apply_peft_adapter",
        return_value=mock_peft_model
    )

    # Mock Trainer
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.5, "eval_accuracy": 0.9} # Return some dummy results
    mock_trainer_class = mocker.patch.object(
        src.trainer,
        "Trainer",
        return_value=mock_trainer_instance
    )

    # Return all the mocks so we can inspect them
    return {
        "load_data": mock_load_data,
        "load_model": mock_load_model,
        "apply_peft": mock_apply_peft,
        "trainer_class": mock_trainer_class,
        "trainer_instance": mock_trainer_instance,
        "wandb_init": mock_wandb_init,
        "mock_run": mock_run 
    }

def test_run_experiment_wiring(mock_dependencies):
    """
    Validates the 'wiring' of run_experiment.
    Checks if all our helper functions are called with the
    correct arguments from the config.
    """
    # Import locally inside the test function
    from src.trainer import run_experiment

    print("\n--- Testing experiment runner wiring ---")

    # Run the function. All heavy parts are mocked.
    peft_model_result, trainer_result = run_experiment(DUMMY_CONFIG, HF_TOKEN)

    # --- Strict Validation ---

    # 1. Was W&B initialized correctly?
    mock_dependencies["wandb_init"].init.assert_called_with(
        project="test_project",
        job_type="sst2_distilbert_lora",
        config=DUMMY_CONFIG,
        name="test_run_lora_r8",
    )

    # 2. Was data loaded correctly? (Check calls for train and validation)
    mock_dependencies["load_data"].assert_any_call(
        task_name="sst2", split="train", token=HF_TOKEN
    )
    # Check validation call for sst2
    mock_dependencies["load_data"].assert_any_call(
        task_name="sst2", split="validation", token=HF_TOKEN
    )


    # 3. Was model loaded correctly?
    mock_dependencies["load_model"].assert_called_with(
        model_name="distilbert", task_name="sst2", token=HF_TOKEN
    )

    # 4. Was PEFT applied correctly?
    mock_dependencies["apply_peft"].assert_called_with(
        mock_dependencies["load_model"].return_value[0], # the base model
        DUMMY_CONFIG
    )
    # Check if parameter counts were logged to wandb summary
    mock_run = mock_dependencies["mock_run"]
    assert mock_run.summary["trainable_params"] > 0
    assert mock_run.summary["total_params"] > 0
    assert mock_run.summary["trainable_percent"] > 0


    # 5. Was Trainer initialized correctly?
    assert mock_dependencies["trainer_class"].called
    trainer_args = mock_dependencies["trainer_class"].call_args[1] # Get kwargs

    assert trainer_args["model"] == mock_dependencies["apply_peft"].return_value # Check model arg
    assert trainer_args["train_dataset"] is not None # Check train dataset was passed (actual value is mock)
    assert trainer_args["eval_dataset"] is not None # Check eval dataset was passed
    assert "compute_metrics" in trainer_args # Check if our func was passed
    assert trainer_args["args"].output_dir == "./results/test_run_lora_r8" # Check TrainingArguments output dir


    # 6. Was training and evaluation run?
    mock_dependencies["trainer_instance"].train.assert_called_once()
    mock_dependencies["trainer_instance"].evaluate.assert_called_once()
    # Check if eval results were logged to wandb summary
    assert mock_run.summary["eval_loss"] == 0.5
    assert mock_run.summary["eval_accuracy"] == 0.9


    # 7. Was the adapter saved correctly?
    mock_peft_model = mock_dependencies["apply_peft"].return_value
    mock_peft_model.save_pretrained.assert_called_with("./results/test_run_lora_r8/best_adapter")

    # 8. Did wandb finish?
    mock_run.finish.assert_called_once()

    print("--- Wiring test passed! ---")