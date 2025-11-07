# LoRA vs. IA³: A Comparative Analysis of PEFT on Diverse Architectures

![Status](https://img.shields.io/badge/Status-Research_Complete-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Transformers%20%7C%20PEFT-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)


This repository contains the code and findings for a deep-dive research project comparing two leading Parameter-Efficient Fine-Tuning (PEFT) techniques: **LoRA (Low-Rank Adaptation)** and **IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)**.

The goal was not to find a single "best" method, but to understand *when* and *why* one might be superior to the other. We tested both methods across all three primary transformer architectures:

  * **Encoder-Only:** `distilbert-base-uncased`
  * **Encoder-Decoder:** `t5-small`
  * **Decoder-Only:** `EleutherAI/pythia-2.8b`

-----

## TL;DR: The Core Finding

After 12+ experiments, our data reveals a clear and powerful conclusion:

> **LoRA and IA³ are not direct competitors—they are different tools for different jobs.**
>
>   * **IA³ (Rescaling)** proved to be the most effective and efficient method for **refining existing skills**. On our classification task, it achieved the best performance with 100x fewer parameters than LoRA.
>
>   * **LoRA (Adding)** was the clear winner for **teaching a model new, complex skills**. On both of our generative tasks (summarization and instruction-following), LoRA dominated IA³ and showed clear performance scaling as we added more parameters (i.e., increased the rank `r`).

-----

## 📊 Final Results Dashboard

This table summarizes the final `eval_loss` (lower is better) for all 12 successful experiments.

| **Model** | **Task** | **PEFT Method** | **Rank (r)** | **Epochs** | **Learning Rate** | **Status** | **Eval Loss** | **Trainable Params (%)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DistilBERT** | `sst2` | **IA³** | **N/A** | 3 | 5e-4 | **Completed** | **🏆 0.3196** | **0.01%** |
| `distilbert` | `sst2` | `lora` | 8 | 3 | 5e-4 | **Completed** | 0.3594 | 1.09% |
| `distilbert` | `sst2` | `lora` | 16 | 3 | 5e-4 | **Completed** | 0.3603 | 1.31% |
| | | | | | | | | |
| **T5** | `samsum` | **LoRA** | **128** | 3 | 5e-4 | **Completed** | **🏆 0.3624** | **7.23%** |
| `t5` | `samsum` | `lora` | 64 | 3 | 5e-4 | **Completed** | 0.3689 | \~3.86% |
| `t5` | `samsum` | `lora` | 32 | 3 | 5e-4 | **Completed** | 0.3761 | \~1.93% |
| `t5` | `samsum` | `lora` | 16 | 3 | 5e-4 | **Completed** | 0.3819 | 0.965% |
| `t5` | `samsum` | `lora` | 8 | 3 | 5e-4 | **Completed** | 0.3880 | 0.485% |
| `t5` | `samsum` | `ia3` | N/A | 3 | 5e-4 | **Completed** | 0.4652 | 0.071% |
| | | | | | | | | |
| **Pythia** | `dolly` | **LoRA** | **128** | 1 | 2e-5 | **Completed** | **🏆 1.9838** | **3.99%** |
| `pythia` | `dolly` | `lora` | 32 | 1 | 2e-5 | **Completed** | 2.0040 | 1.03% |
| `pythia` | `dolly` | `lora` | 8 | 1 | 2e-5 | **Completed** | 2.0333 | 0.26% |
| `pythia` | `dolly` | `ia3` | N/A | 1 | 2e-5 | **Completed** | 2.2410 | 0.024% |

-----

## 🔬 In-Depth Analysis & Key Insights

Our findings show a clear pattern based on task complexity.

### 1\. Experiment 1: DistilBERT (Encoder-Only / Classification)

  * **Task:** Simple sentiment classification (`sst2`). The base model *already* understands English and sentiment. It just needs to be "focused" on the task.
  * **Winner:** **IA³** (Loss 0.3196).
  * **Analysis:** IA³ was the clear winner. It achieved a better loss than LoRA while training *100x fewer* parameters (0.01% vs 1.09%). This supports our hypothesis: for a simple task where the knowledge *already exists*, rescaling (IA³) is a far more efficient and effective strategy than adding new weights (LoRA).

### 2\. Experiment 2: T5 (Encoder-Decoder / Generation)

  * **Task:** Abstractive summarization (`samsum`). This is a *new, complex skill* the model needs to learn.
  * **Winner:** **LoRA `r=128`** (Loss 0.3624).
  * **Analysis:** LoRA *dominated*. IA³ (Loss 0.4652) was completely non-competitive. Furthermore, LoRA's performance scaled directly with its size: `r=128 > r=64 > r=32 > r=16 > r=8`. To learn this *new skill*, the model *needed new parameters*. IA³'s "rescaling" was insufficient.

### 3\. Experiment 3: Pythia (Decoder-Only / Generation)

  * **Task:** Instruction-following (`dolly-15k`). Like summarization, this is a *new, complex skill*.
  * **Winner:** **LoRA `r=128`** (Loss 1.9838).
  * **Analysis:** This experiment **perfectly confirms the T5 results**. LoRA was the clear winner, beating IA³ (Loss 2.2410) by a significant margin. And just like with T5, performance scaled with rank: `r=128 > r=32 > r=8`. This finding is robust across two different generative architectures.

-----

## 🛠️ Methodology & Debugging Journey

A key part of this project was overcoming numerical instability when training large models on consumer GPUs.

### The Gemma-to-Pythia Pivot

Initial experiments on `gemma-2b` (our first choice for a Decoder-Only model) failed catastrophically. All training runs resulted in an `eval_loss` of `~128.5`, a clear sign of the loss exploding to `NaN` on the first step.

This led to a deep investigation into the `fp16` stability stack on a T4 GPU. We discovered that the combination of `gemma-2b` + `load_in_8bit=True` + `fp16=True` was fundamentally unstable, failing even with gradient clipping and 8-bit optimizers.

To isolate the variable, we pivoted to **`EleutherAI/pythia-2.8b`**, a stable, research-focused model of a similar scale. We then built a stable training configuration from scratch (see `notebooks/Run_Pythia_Dolly.ipynb`), which proved our methodology was sound and that the instability was model-specific.

**Our Final Stable Training Stack (for T4 GPUs):**

  * `load_in_8bit=True` (from `bitsandbytes`)
  * `fp16=True` (for mixed-precision)
  * `optim="paged_adamw_8bit"` (for 8-bit stable optimization)
  * `max_grad_norm=0.3` (for gradient clipping)
  * `learning_rate=2e-5` (a safe, low LR)

-----

## 📂 Project Structure

```
lora-vs-ia3/
│
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── Run_DistilBERT_SST2.ipynb   # (Uses src package)
│   ├── Run_T5_SAMSum.ipynb         # (Uses src package)
│   └── Run_Pythia_Dolly.ipynb      # (Standalone, no src)
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── metrics.py
    ├── model_loader.py
    ├── peft_utils.py
    └── trainer.py
```

-----

## 🚀 How to Run the Experiments

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR-USERNAME/lora-vs-ia3.git
    cd lora-vs-ia3
    ```
2.  **Create a virtual environment** and install requirements:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Install the `src` package** in editable mode:
    ```bash
    pip install -e .
    ```
4.  **Create `.env` file:** Create a `.env` file in the root directory and add your Hugging Face token:
    ```
    HF_TOKEN=hf_YOUR_TOKEN_HERE
    ```

### Experiment 1 & 2 (DistilBERT & T5)

These experiments use the `src` package.

1.  Launch Jupyter and open `notebooks/Run_DistilBERT_SST2.ipynb`.
2.  Set your W\&B API key and run all cells.
3.  Repeat for `notebooks/Run_T5_SAMSum.ipynb`.

### Experiment 3 (Pythia)

This experiment is **standalone** and does not depend on the `src` package. It was designed this way to rigorously test and isolate the numerical instability we faced.

1.  Open `notebooks/Run_Pythia_Dolly.ipynb` in Google Colab.
2.  Set the runtime to T4 GPU.
3.  Add your `HF_TOKEN` and `WANDB_API_KEY` to the Colab Secrets manager.
4.  Run all cells from top to bottom.

-----

## 🔮 Future Work

**Due to significant compute constraints (training on a single T4 GPU), several methodological trade-offs were made:**

  * **Training & Evaluation on Subsets:** All generative models (T5, Pythia) were trained and evaluated on small subsets of the full dataset (e.g., 5k train / 200 eval) to prevent CPU RAM OOM errors.
  * **Saved Adapters:** The final trained adapters for the Pythia model were not saved due to session limitations.

This provides clear paths for future work:

1.  **Re-run the final Pythia experiments** on a more powerful machine and save the adapters.
2.  **Perform Quantitative Benchmarking:** The original plan included benchmarking generative text with metrics like ROUGE and Perplexity. This was disabled during training to prevent RAM crashes. The saved adapters would allow for a separate, post-training benchmarking script to be run.
3.  **Perform Qualitative Benchmarking:** With saved adapters, a "Human-in-the-Loop" evaluation could be performed by generating responses for 20-30 identical prompts and scoring them for "coherence" and "helpfulness."
