# src/trainer.py

import os
import wandb
from functools import partial
from transformers import (
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from src.data_loader import load_and_preprocess_dataset
from src.model_loader import load_base_model, MODEL_NAME_MAP
from src.peft_utils import apply_peft_adapter
from src.metrics import compute_metrics

def run_experiment(config: dict, token: str):
    """
    Runs a single fine-tuning experiment based on the provided config.
    """
    
    # --- 1. Initialize W&B ---
    job_type = f"{config['task_name']}_{config['model_name']}_{config['peft_method']}"
    run = wandb.init(
        project=config.get("wandb_project", "lora-vs-ia3"),
        job_type=job_type,
        config=config,
        name=config.get("run_name"),
    )

    # --- 2. Load Data Subsets (Unchanged) ---
    print(f"--- Loading dataset: {config['task_name']} ---")
    tokenizer_name = MODEL_NAME_MAP[config['model_name']]

    train_subset_size = config.get("train_subset_size", None)
    if train_subset_size:
        print(f"--- Loading SUBSET of {train_subset_size} training samples ---")
        full_train_dataset = load_and_preprocess_dataset(
            task_name=config['task_name'], split='train', token=token
        )
        train_subset_size = min(train_subset_size, len(full_train_dataset))
        train_indices = range(train_subset_size)
        train_dataset = full_train_dataset.select(train_indices)
        print(f"Using {len(train_dataset)} training samples.")
    else:
        print("--- Loading FULL training dataset ---")
        train_dataset = load_and_preprocess_dataset(
            task_name=config['task_name'], split='train', token=token
        )

    eval_split_name = 'validation'
    eval_subset_size = config.get("eval_subset_size", 100)
    print(f"--- Loading SUBSET of {eval_subset_size} evaluation samples ---")

    if config['task_name'] == 'dolly':
        print("--- Splitting Dolly train/eval subset ---")
        full_train_for_split = load_and_preprocess_dataset(
            task_name=config['task_name'], split='train', token=token
        )
        total_size = len(full_train_for_split)
        if eval_subset_size >= total_size * 0.5:
            eval_subset_size = int(total_size * 0.1)
        if train_subset_size is None or (train_subset_size + eval_subset_size) > total_size:
            train_subset_size = total_size - eval_subset_size
        split_dataset = full_train_for_split.train_test_split(test_size=eval_subset_size, seed=42)
        train_indices = range(min(train_subset_size, len(split_dataset['train'])))
        train_dataset = split_dataset['train'].select(train_indices)
        eval_dataset_subset = split_dataset['test']
        print(f"Using {len(train_dataset)} train and {len(eval_dataset_subset)} eval samples for Dolly.")

    elif config['task_name'] in ['sst2', 'samsum']:
        full_eval_dataset = load_and_preprocess_dataset(
            task_name=config['task_name'],
            split=eval_split_name,
            token=token
        )
        eval_subset_size = min(eval_subset_size, len(full_eval_dataset))
        eval_indices = range(eval_subset_size)
        eval_dataset_subset = full_eval_dataset.select(eval_indices)
        print(f"Using {len(eval_dataset_subset)} eval samples out of {len(full_eval_dataset)}.")
    else:
         raise ValueError(f"Unhandled task for validation split: {config['task_name']}")

    # --- 3. Load Model (Unchanged) ---
    print(f"--- Loading model: {config['model_name']} ---")
    base_model, tokenizer = load_base_model(
        model_name=config['model_name'],
        task_name=config['task_name'],
        token=token
    )

    # --- 4. Apply PEFT Adapter (Unchanged) ---
    print(f"--- Applying PEFT adapter: {config['peft_method']} ---")
    peft_model = apply_peft_adapter(base_model, config)
    peft_model.print_trainable_parameters()
    # (W&B logging of params...)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    run.summary["trainable_params"] = trainable_params
    run.summary["total_params"] = total_params
    run.summary["trainable_percent"] = (trainable_params / total_params) * 100 if total_params > 0 else 0


    # --- 5. Data Collator ---
    if config['task_name'] == "samsum":
        print("--- Using DataCollatorForSeq2Seq ---")
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=peft_model)
    elif config['task_name'] == "dolly":
        print("--- Using DataCollatorForLanguageModeling (Causal LM) ---")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else: # This is for 'sst2'
        print("--- Using DataCollatorWithPadding (Classification) ---")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 6. Metrics Function (Unchanged) ---
    compute_metrics_partial = partial(
        compute_metrics,
        task_name=config['task_name'],
        tokenizer_name=tokenizer_name,
        token=token
    )

    metric_to_optimize = "eval_loss"
    if config['task_name'] == 'sst2':
        metric_to_optimize = "eval_accuracy"

    output_dir = f"./results/{config.get('run_name', 'experiment')}"


    # Define common arguments for both trainers
    common_args = {
        "output_dir": output_dir,
        "num_train_epochs": config.get("num_epochs", 3),
        "per_device_train_batch_size": config.get("batch_size", 4),
        "per_device_eval_batch_size": 2, # Keep eval batch size small
        "gradient_accumulation_steps": config.get("gradient_accumulation", 1),
        "warmup_steps": config.get("warmup_steps", 50),
        "learning_rate": config.get("learning_rate", 5e-5),
        "weight_decay": 0.01,
        "logging_dir": f"{output_dir}/logs",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_to_optimize,
        "greater_is_better": metric_to_optimize != "eval_loss",
        "report_to": "wandb",
        "fp16": True,
        "bf16": False,
        "max_grad_norm": 0.3,
        "optim": "paged_adamw_8bit",
    }

    if config['task_name'] == "samsum": # <-- WAS: or config['task_name'] == "dolly"
        print("--- Using Seq2SeqTrainer (for eval_loss only) ---")
        training_args = Seq2SeqTrainingArguments(
            **common_args
        )
        TrainerClass = Seq2SeqTrainer
        metrics_fn_to_pass = None # Disable ROUGE metrics during training
    
    else: # This is for 'sst2' AND 'dolly'
        print("--- Using standard Trainer ---")
        training_args = TrainingArguments(
            **common_args
        )
        TrainerClass = Trainer
        # Only compute metrics for sst2
        metrics_fn_to_pass = compute_metrics_partial if config['task_name'] == 'sst2' else None
    # --- END FIX ---
 


    # --- 7. Initialize Trainer ---
    trainer = TrainerClass(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_subset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_fn_to_pass,
    )

    # --- 8. Train and Evaluate ---
    print(f"--- Starting Training: {config.get('run_name')} ---")
    trainer.train()
    
    print(f"--- Starting Evaluation (calculating eval_loss) ---")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    run.summary.update(eval_results)

    # --- 9. Save and Finish ---
    print(f"--- Saving Adapter ---")
    adapter_path = f"{output_dir}/best_adapter"
    peft_model.save_pretrained(adapter_path)
    print(f"Best adapter saved to {adapter_path}")
    
    run.finish()
    
    return peft_model, trainer