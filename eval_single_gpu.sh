#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_single_gpu.sh — Run EviNote-RAG evaluation on a single 4090 GPU
#
# Two steps, each in its own terminal:
#   Terminal 1:  Start BM25 retriever   (conda activate retriever)
#   Terminal 2:  Run evaluation          (conda activate EviNoteRAG)
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths (edit these to match your setup) ───────────────────────────────────
BM25_INDEX="/finder/qyj/sparse_retriever/bm25"
CORPUS="/finder/qyj/sparse_retriever/wiki-18.jsonl"
MODEL="/finder/qyj/models/EviNoteRAG-7B"

# ── Step 1: Start the BM25 retriever (run in Terminal 1) ────────────────────
# conda activate retriever
# python RAG/search/retrieval_server.py \
#     --retriever_name bm25 \
#     --index_path $BM25_INDEX \
#     --corpus_path $CORPUS \
#     --topk 3

# ── Step 2: Run evaluation (run in Terminal 2) ──────────────────────────────
# conda activate EviNoteRAG
python eval_single_gpu.py \
    --model_id $MODEL \
    # --num_samples 200 \
    --max_turns 4 \
    --topk 3 \
    --retriever_url "http://127.0.0.1:8000/retrieve" \
    --output_dir ./outputs/eval_single_gpu

# ── Notes ────────────────────────────────────────────────────────────────────
# - BM25 retriever is CPU-only, so the full GPU is available for the 7B model.
# - At ~30-60s per question, 200 questions ≈ 2-3 hours.
# - Results are saved incrementally; use --resume_from N to continue after crash.
# - To run more:  --num_samples 500  (or remove for the full 17k val set)
# - Output: outputs/eval_single_gpu/triviaqa_results.jsonl (per-question details)
#           outputs/eval_single_gpu/triviaqa_summary.json   (aggregate F1/EM)
