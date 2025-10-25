# src/peft_utils.py

from peft import get_peft_model, LoraConfig, IA3Config, TaskType
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import re

# This is the trickiest part. The "target_modules" (which layers to adapt)
# have different names for different models. This dict maps model types
# to their layer names.

PEFT_TARGET_MODULES_MAP = {
    # DistilBERT
    "DistilBertForSequenceClassification": {
        "lora": ["q_lin", "v_lin"],
        "ia3": ["q_lin", "v_lin", "out_lin"], 
    },
    # T5
    "T5ForConditionalGeneration": {
        "lora": ["q", "v"],
        "ia3": ["k", "v", "wo"], # As per the IA3 paper
    },
    # Gemma
    "GemmaForCausalLM": {
        "lora": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "ia3": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"], 
    },
}

# This maps our task names to the PEFT library's TaskType enum
TASK_TYPE_MAP = {
    "sst2": TaskType.SEQ_CLS,
    "samsum": TaskType.SEQ_2_SEQ_LM,
    "dolly": TaskType.CAUSAL_LM,
}

def apply_peft_adapter(model, peft_config: dict):
    """
    Applies a PEFT adapter to a base model based on a config dictionary.

    Args:
        model: The base Hugging Face model.
        peft_config (dict): A dictionary specifying the PEFT method and its
                           hyperparameters. Must include 'peft_method'.

    Returns:
        A PeftModel object.
    """
    peft_method = peft_config.get("peft_method")
    if not peft_method:
        raise ValueError("peft_config must include 'peft_method' ('lora' or 'ia3')")

    # 1. Get model class name (e.g., "T5ForConditionalGeneration")
    model_class_name = model.__class__.__name__
    if model_class_name not in PEFT_TARGET_MODULES_MAP:
        raise ValueError(f"Unknown model class: {model_class_name}. No target modules defined.")

    # 2. Get the task type
    task_name = peft_config.get("task_name")
    if not task_name or task_name not in TASK_TYPE_MAP:
        raise ValueError(f"peft_config must include 'task_name' {list(TASK_TYPE_MAP.keys())}")
    task_type = TASK_TYPE_MAP[task_name]

    # 3. Get target modules for this model and method
    target_modules = PEFT_TARGET_MODULES_MAP[model_class_name].get(peft_method)
    if not target_modules:
        raise ValueError(f"No target modules defined for method '{peft_method}' on model '{model_class_name}'")

    # 4. Build the PEFT config object
    if peft_method == "lora":
        config = LoraConfig(
            task_type=task_type,
            r=peft_config.get("r", 8),
            lora_alpha=peft_config.get("lora_alpha", 16),
            lora_dropout=peft_config.get("lora_dropout", 0.1),
            target_modules=target_modules,
            bias="none",
        )
    
    elif peft_method == "ia3":
        # IA3 splits targets into attention and feedforward modules
        feedforward_modules = []
        
        # This regex finds typical feedforward/MLP layer names
        ff_pattern = r".*(mlp|fc|wi|w0|w1|w2|down_proj|out_lin).*" 
        
        attention_modules = [m for m in target_modules if not re.match(ff_pattern, m)]
        feedforward_modules = [m for m in target_modules if re.match(ff_pattern, m)]

        if not feedforward_modules:
             # This is a fallback to ensure we don't crash
             print(f"Warning: No feedforward modules found for IA3 on {model_class_name}. Using attention modules only.")
             
        config = IA3Config(
            task_type=task_type,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
        )
        
    else:
        raise ValueError(f"Unknown peft_method: {peft_method}")

    # 5. Apply the adapter to the model
    peft_model = get_peft_model(model, config)
    return peft_model