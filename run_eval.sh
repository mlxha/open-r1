#!/bin/bash
set -e

USERNAME="${LOGNAME}"

# Auto-detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    TOTAL_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "=== Detected $TOTAL_GPUS GPUs ==="
    
    if [ "$TOTAL_GPUS" -eq 0 ]; then
        echo "ERROR: No GPUs detected. Exiting."
        exit 1
    fi
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

if [ -z "$1" ]; then
    echo "ERROR: Model name must be provided as the first argument."
    echo "Usage: bash run_eval.sh <model_name>"
    exit 1
fi
MODEL="$1"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$TOTAL_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

WORKSPACE_DIR="/workspace"
mkdir -p "$WORKSPACE_DIR/env"
cd "/lightscratch/users/$USERNAME/open-r1/"

# Setup environment
echo "=== Environment Setup in $WORKSPACE_DIR ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_ROOT="$WORKSPACE_DIR/env/.uv"
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.local/bin/env"

echo "=== Virtual Environment ==="
uv venv "$WORKSPACE_DIR/env/openr1" --python 3.11
source "$WORKSPACE_DIR/env/openr1/bin/activate"

echo "=== Dependency Installation ==="
uv pip install --upgrade pip
make install

# Log in to Hugging Face using the API key file
if [ -f "$HF_API_KEY_FILE_AT" ]; then
    echo "=== Logging in to Hugging Face ==="
    huggingface-cli login --token "$(cat "$HF_API_KEY_FILE_AT")"
else
    echo "=== Hugging Face API key file not found at $HF_API_KEY_FILE_AT ==="
    exit 1
fi
# Log in to Weights & Biases using the API key file
if [ -f "$WANDB_API_KEY_FILE_AT" ]; then
    echo "=== Logging in to Weights & Biases ==="
    wandb login "$(cat "$WANDB_API_KEY_FILE_AT")"
else
    echo "=== WandB API key file not found at $WANDB_API_KEY_FILE_AT ==="
    exit 1
fi

# Run benchmarks
TASK=pubmedqa
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/custom_tasks.py \
    --use-chat-template \
    --output-dir evals/${MODEL}/${TASK}

TASK=medmcqa
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/custom_tasks.py \
    --use-chat-template \
    --output-dir evals/${MODEL}/${TASK}

TASK=medqa
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/custom_tasks.py \
    --use-chat-template \
    --output-dir evals/${MODEL}/${TASK}

TASK=medmcqa_gen
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/custom_tasks.py \
    --use-chat-template \
    --output-dir evals/${MODEL}/${TASK}

TASK=medqa_gen
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/custom_tasks.py \
    --use-chat-template \
    --output-dir evals/${MODEL}/${TASK}

TASK=aime24
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir evals/${MODEL}/${TASK}

TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir evals/${MODEL}/${TASK}