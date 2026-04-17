#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline C: Claim-level evidence organization evaluation
# Compare results against Pipeline B (eval_single_gpu.py with base model)
# ─────────────────────────────────────────────────────────────────────────────

MODEL="/finder/qyj/models/Qwen2.5-7B-Instruct"

export HF_ENDPOINT="https://hf-mirror.com"

python eval_single_gpu_claimlevel.py \
    --model_id $MODEL \
    --num_samples 200 \
    --max_turns 4 \
    --topk 3 \
    --retriever_url "http://127.0.0.1:8000/retrieve" \
    --output_dir ./outputs/eval_claimlevel_v2
