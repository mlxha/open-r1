# Open R1

*A fully open reproduction of DeepSeek-R1. This repo is work in progress, let's build it together!*

## Overview

The goal of this repo is to build the missing pieces of the R1 pipeline such that everybody can reproduce and build on top of it. The project is simple by design and mostly consists of:

- `src/open_r1` contains the scripts to train and evaluate models as well generate synthetic data:
    - `grpo.py`: trains a model with GRPO on a given dataset
    - `sft.py`: simple SFT of a model on a dataset
    - `evaluate.py`: evaluates a model on the R1 benchmarks
    - `generate`: contains the Slurm and Distilabel scripts to generate synthetic data with a model
- `Makefile` contains an easy to run command for each step in the R1 pipeline leveraging the scipts above.

### Plan of attack

We will use the DeepSeek-R1 [tech report](https://github.com/deepseek-ai/DeepSeek-R1) as a guide, which can roughly be broken down into three main steps:

* Step 1: replicate the R1-Distill models by distilling a high-quality corpus from DeepSeek-R1.
* Step 2: replicate the pure RL pipeline that DeepSeek used to create R1-Zero. This will likely involve curatint new, large-scale datasets for math, reasoning, and code.
* Step 3: show we can go from base model to RL-tuned via multi-stage training.

<center>
    <img src="assets/plan-of-attack.png" width="500">
</center>


## Installation

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n openr1 python=3.11 && conda activate openr1
```

Next, install vLLM:

```shell
pip install vllm==0.6.6.post1

# For HF (cluster only has CUDA 12.1)
pip install vllm==0.6.6.post1 --extra-index-url https://download.pytorch.org/whl/cu121
```

This will also install PyTorch `v2.5.1` and it is **very important** to use this version since the vLLM binaries are compiled for it. You can then install the remaining dependencies for your specific use case via `pip install -e .[LIST OF MODES]`. For most contributors, we recommend:

```shell
pip install -e ".[dev]"
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, check your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

```shell
git-lfs --version
```

If it isn't installed, run:

```shell
sudo apt-get install git-lfs
```

## Training models

### SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k), use this command, or edit `launch.slurm`.

```
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Distill-R1
```

### GRPO

```
accelerate launch src/open_r1/grpo.py \
    --output_dir Qwen2.5-0.5B-GRPO \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name AI-MO/NuminaMath-TIR \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10
```

## Evaluating models

For small models use `--data_parallel=$NUM_GPUS`, for large models shard with `--tensor_parallel=$NUM_GPUS`
Example for evaluating `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B `

```
NUM_GPUS=1
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24 # or math_500
OUTPUT_DIR=data/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" --use-chat-template --custom-tasks src/open_r1/evaluate.py --output-dir $OUTPUT_DIR --system-prompt="Please reason step by step, and put your final answer within \boxed{}."
```

