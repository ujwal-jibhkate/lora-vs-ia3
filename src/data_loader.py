# src/data_loader.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# A dictionary to map our internal task names to their Hugging Face names
DATASET_CONFIGS = {
    "sst2": {
        "hf_name": "glue",
        "hf_subset": "sst2",
        "model_tokenizer_name": "distilbert-base-uncased",
    },
    "samsum": {
        "hf_name": "knkarthick/samsum",  # <-- FIX 1: Updated path
        "hf_subset": None,              # <-- FIX 1: This is now correct
        "model_tokenizer_name": "t5-small",
    },
    "dolly": {
        "hf_name": "databricks/databricks-dolly-15k",
        "hf_subset": None,
        "model_tokenizer_name": "google/gemma-2b",
    },
}


def load_and_preprocess_dataset(task_name: str, split: str = "train", token: str = None): # <-- FIX 2: Added token
    """
    Loads, preprocesses, and tokenizes a dataset for a specific task.

    Args:
        task_name (str): One of 'sst2', 'samsum', or 'dolly'.
        split (str): The dataset split to load (e.g., 'train', 'validation').
        token (str, optional): Hugging Face token for gated models.
    """
    if task_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown task name: {task_name}. Must be one of {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[task_name]

    # 1. Load tokenizer
    # We load the *specific* tokenizer for the model we will use
    # for this task. This is critical.
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_tokenizer_name"], 
        token=token  # <-- FIX 2: Pass token
    )
    
    # Handle padding for T5 vs. other models
    if "t5" in config["model_tokenizer_name"]:
        tokenizer.pad_token = tokenizer.eos_token
    elif "gemma" in config["model_tokenizer_name"]:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset
    if config["hf_subset"]:
        dataset = load_dataset(
            config["hf_name"], 
            config["hf_subset"], 
            split=split, 
            token=token  # <-- FIX 2: Pass token
        )
    else:
        dataset = load_dataset(
            config["hf_name"], 
            split=split, 
            token=token  # <-- FIX 2: Pass token
        )

    # 3. Define task-specific preprocessing functions
    def preprocess_sst2(examples):
        # (Rest of this function is unchanged)
        tokenized_inputs = tokenizer(
            examples["sentence"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        tokenized_inputs["labels"] = examples["label"]
        return tokenized_inputs

    def preprocess_samsum(examples):
        # (Rest of this function is unchanged)
        prompt = "summarize: "
        inputs = [prompt + doc for doc in examples["dialogue"]]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=512, 
            truncation=True, 
            padding="max_length"
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"], 
                max_length=128, 
                truncation=True, 
                padding="max_length"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_dolly(examples):
        # (Rest of this function is unchanged)
        prompts = []
        for instruction, context, response in zip(examples['instruction'], examples['context'], examples['response']):
            text = "Instruction:\n"
            text += f"{instruction}\n"
            if context:
                text += "Context:\n"
                text += f"{context}\n"
            text += "Response:\n"
            text += f"{response}"
            prompts.append(text)
            
        tokenized_inputs = tokenizer(
            prompts, 
            truncation=True, 
            padding="max_length", 
            max_length=512 
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
        return tokenized_inputs

    # 4. Apply the correct preprocessing
    if task_name == "sst2":
        # (This section is unchanged)
        tokenized_dataset = dataset.map(preprocess_sst2, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
    elif task_name == "samsum":
        # (This section is unchanged)
        tokenized_dataset = dataset.map(preprocess_samsum, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["id", "dialogue", "summary"])
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    elif task_name == "dolly":
        # (This section is unchanged)
        tokenized_dataset = dataset.map(preprocess_dolly, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["instruction", "context", "response", "category"])
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
    else:
        raise RuntimeError("Invalid task name logic.")

    return tokenized_dataset