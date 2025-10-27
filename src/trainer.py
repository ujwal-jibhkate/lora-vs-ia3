# src/trainer.py

import os
import wandb
from functools import partial
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
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

    # --- 2. Load Data, Model, Tokenizer ---
    print(f"--- Loading dataset: {config['task_name']} ---")
    tokenizer_name = MODEL_NAME_MAP[config['model_name']]

    train_dataset = load_and_preprocess_dataset(
        task_name=config['task_name'],
        split='train',
        token=token
    )
    # Correctly load the validation split based on task
    validation_split = 'validation'
    if config['task_name'] == 'dolly':
        # Dolly doesn't have a standard validation split, use a slice of train
        # Note: In a real scenario, you'd create a proper split beforehand
        print("--- Using train split slice for Dolly validation ---")
        train_eval_split = train_dataset.train_test_split(test_size=0.1) # Use 10% for eval
        train_dataset = train_eval_split['train']
        eval_dataset = train_eval_split['test']
    elif config['task_name'] == 'sst2':
         eval_dataset = load_and_preprocess_dataset(
            task_name=config['task_name'],
            split='validation', # SST2 has a validation split
            token=token
        )
    elif config['task_name'] == 'samsum':
        eval_dataset = load_and_preprocess_dataset(
            task_name=config['task_name'],
            split='validation', # Samsum has a validation split
            token=token
        )
    else:
         raise ValueError(f"Unhandled task for validation split: {config['task_name']}")


    print(f"--- Loading model: {config['model_name']} ---")
    base_model, tokenizer = load_base_model(
        model_name=config['model_name'],
        task_name=config['task_name'],
        token=token
    )

    # --- 3. Apply PEFT Adapter ---
    print(f"--- Applying PEFT adapter: {config['peft_method']} ---")
    peft_model = apply_peft_adapter(base_model, config)
    peft_model.print_trainable_parameters()

    # Log parameter count to W&B
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters()) # Use the PEFT model here too
    run.summary["trainable_params"] = trainable_params
    run.summary["total_params"] = total_params
    run.summary["trainable_percent"] = (trainable_params / total_params) * 100 if total_params > 0 else 0


    # --- 4. Define Data Collator ---
    if config['task_name'] == "samsum":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=peft_model
        )
    else:
        # Includes sst2 and dolly (causal LM uses padding too)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 5. Define Metrics Function ---
    compute_metrics_partial = partial(
        compute_metrics,
        task_name=config['task_name'],
        tokenizer_name=tokenizer_name,
        token=token
    )

    # --- 6. Define Training Arguments ---
    output_dir = f"./results/{config.get('run_name', 'experiment')}"

    # Determine metric for best model based on task
    metric_to_optimize = "eval_loss" # Default (good for dolly, samsum)
    if config['task_name'] == 'sst2':
        metric_to_optimize = "eval_accuracy" # Or eval_f1

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=config.get("gradient_accumulation", 1), # Added for flexibility
        warmup_steps=config.get("warmup_steps", 50),
        learning_rate=config.get("learning_rate", 5e-5), # Added learning rate config
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps", # Log more frequently
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_to_optimize,
        greater_is_better=metric_to_optimize != "eval_loss", # Accuracy/F1 higher is better
        report_to="wandb",
        fp16=True, # Use mixed precision if GPU supports it
        bf16=False, # Could also use bf16 if GPU supports it
        # deepspeed=config.get("deepspeed_config_path", None), # Optional DeepSpeed config
    )

    # --- 7. Initialize Trainer ---
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_partial if config['task_name'] != 'dolly' else None, # Only pass if needed
    )

    # --- 8. Train and Evaluate ---
    print(f"--- Starting Training: {config.get('run_name')} ---")
    trainer.train()

    print(f"--- Starting Evaluation: {config.get('run_name')} ---")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    # Log final eval results explicitly to W&B summary
    run.summary.update(eval_results)

    # --- 9. Save and Finish ---
    adapter_path = f"{output_dir}/best_adapter"
    peft_model.save_pretrained(adapter_path)
    print(f"Best adapter saved to {adapter_path}")

    run.finish()

    return peft_model, trainer
