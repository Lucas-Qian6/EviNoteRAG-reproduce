# Offline Re-Scorer

Re-scores existing claim-level eval outputs without re-running the model.
The original `extract_answer` in the eval scripts has two failure modes that
together account for most of the unexpectedly low NQ / HotpotQA scores:

1. **Empty-tag mimic (Qwen2.5 + claim prompt).** The prompt contains the
   literal string `<answer></answer>`, so Qwen2.5-Instruct often copies it
   verbatim instead of filling it. The original extractor returns `None` for
   empty tags → counted as `[NO_ANSWER]`. The new extractor only drops empties
   from consideration; the right answer was usually never produced in the first
   place, so this fix mainly lowers measurement noise. The real fix here is the
   prompt template, which still requires a re-run.

2. **Repetition loop (EviNote-RAG-7B + claim prompt).** The RL-trained model
   produces the correct answer once, then collapses into
   `<answer>X</answer><answer></answer><answer>X</answer>...` until it hits
   `max_new_tokens`. The original extractor takes `matches[-1]`, which is
   almost always the trailing empty tag. The new extractor takes the first
   non-empty answer (or majority-vote when there are repeats), correctly
   recovering `X` for these cases.

## Usage

From the repo root:

```bash
# Re-score a single results file (writes <input>_rescored.jsonl alongside it)
python rescorer/rescore.py \
    --input outputs/eval_nq_8gpus_claimlevel/nq_results.jsonl

# Re-score every results file under outputs/ at once
python rescorer/rescore.py --all --summary_json rescorer/summary.json

# Just print before/after numbers without writing output files
python rescorer/rescore.py --input outputs/eval_nq_8gpus_claimlevel/nq_results.jsonl --diff
```

The output JSONL preserves every field of the original record and adds:

- `extracted_answer_old`, `f1_old`, `em_old`: original values
- `extracted_answer`, `f1`, `em`: rescored values (overwritten in place)
- `extract_source`: which extractor branch produced the new answer
  (`first_nonempty_tag`, `majority_vote_tag`, `fallback_prose`, `no_answer`)

## Spot-checking

```bash
# Cases the new extractor recovered from [NO_ANSWER]
python rescorer/inspect_failures.py \
    --input outputs/eval_nq_8gpus_claimlevel_evinote/nq_results_rescored.jsonl \
    --mode recovered --n 5

# Cases that are still NO_ANSWER (these are the ones that need a prompt fix + re-run)
python rescorer/inspect_failures.py \
    --input outputs/eval_nq_8gpus_claimlevel/nq_results_rescored.jsonl \
    --mode still_no_answer --n 5

# Sanity: any case where the new score is *lower* than the old one
python rescorer/inspect_failures.py \
    --input outputs/eval_nq_8gpus_claimlevel_evinote/nq_results_rescored.jsonl \
    --mode regressions --n 5
```

## Expected impact

Based on the failure-mode counts diagnosed earlier:

| run | old F1 | expected new F1 |
|---|---|---|
| Qwen2.5+claim NQ (3610, 21.5% NO_ANSWER) | 0.289 | small bump (most empties were truly empty) |
| EviNote+claim NQ (3610, 6.5% NO_ANSWER, repetition loop) | 0.329 | ~0.40+ (recovers the loop cases) |

Run on your HotpotQA outputs as well — the repetition-loop pattern almost
certainly appears there too.
