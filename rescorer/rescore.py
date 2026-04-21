"""
Offline re-scorer for claim-level evaluation outputs.

The original eval scripts save `full_output` for every question. This script
re-extracts the answer with a more robust extractor and re-computes F1 / EM
without needing a GPU re-run.

Two extractor failure modes this fixes:

1. Qwen2.5 + claim prompt: the model often emits the literal `<answer></answer>`
   from the prompt template (empty tags). The original extractor returns
   `None` for these → counted as 0. The new extractor drops empty matches and
   then takes the first non-empty `<answer>...</answer>` from the model's
   output region, so we at least don't lose anything that *was* answered.

2. EviNote-RAG-7B + claim prompt: the RL-trained model often produces the
   correct answer, then collapses into a repetition loop alternating
   `<answer>X</answer><answer></answer><answer>X</answer>...` until it hits
   max_tokens. The original extractor returns `matches[-1]` which is usually
   the trailing empty tag. The new extractor returns the first non-empty
   answer from the model's output region — for these loops it's always
   the correct one.

Usage:

    # Re-score a single results file
    python rescorer/rescore.py \
        --input outputs/eval_nq_8gpus_claimlevel/nq_results.jsonl \
        --output outputs/eval_nq_8gpus_claimlevel/nq_results_rescored.jsonl

    # Re-score every claim-level results file under outputs/
    python rescorer/rescore.py --all

    # Compare against the existing recorded scores
    python rescorer/rescore.py \
        --input outputs/eval_nq_8gpus_claimlevel/nq_results.jsonl \
        --diff
"""

import argparse
import glob
import json
import os
import re
import string
from collections import Counter


# ── Scoring (identical to the eval scripts) ─────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    max_f1 = 0.0
    for golden_answer in golden_answers:
        gold_tokens = normalize_answer(golden_answer).split()
        if not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        num_same = len(common)
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        max_f1 = max(max_f1, f1)
    return max_f1


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


# ── Robust answer extraction ────────────────────────────────────────────────

ANSWER_RE = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>",
                       re.DOTALL | re.IGNORECASE)
FALLBACK_PATTERNS = [
    re.compile(r"final\s*answer\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
    re.compile(r"\banswer\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
]
# When the prompt mention is captured as the first <answer></answer>, drop it.
# The prompt's mention is always the empty form; the model's real answers come
# later in the text.


def extract_answer_robust(full_output):
    """Return (answer_string, source_tag) or (None, "no_answer").

    source_tag describes where the answer came from for debugging:
      - "first_nonempty_tag": found a non-empty <answer>...</answer>
      - "majority_vote_tag": >=2 distinct non-empty tag answers, took the most
        common (handles the EviNote repetition loop without picking truncated
        partials)
      - "fallback_prose": no usable <answer> tag, fell back to "Answer: ..."
        prose
      - "no_answer": nothing usable
    """
    matches = ANSWER_RE.findall(full_output)
    nonempty = [m.strip() for m in matches if m.strip()]

    if nonempty:
        # If the model repeats the same answer many times (EviNote loop),
        # majority vote on the normalized form is the most robust pick.
        counts = Counter(normalize_answer(a) for a in nonempty)
        top_norm, top_count = counts.most_common(1)[0]
        if top_count >= 2:
            for a in nonempty:
                if normalize_answer(a) == top_norm:
                    return a, "majority_vote_tag"
        return nonempty[0], "first_nonempty_tag"

    for pat in FALLBACK_PATTERNS:
        m = pat.search(full_output)
        if m:
            cand = m.group(1).strip().strip(".").strip('"').strip("'")
            if cand:
                return cand, "fallback_prose"

    return None, "no_answer"


# ── Re-scoring core ─────────────────────────────────────────────────────────

def rescore_file(input_path, output_path=None, diff=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    new_records = []
    n_total = 0
    n_old_no_answer = 0
    n_new_no_answer = 0
    n_recovered = 0
    n_changed_score = 0
    source_counts = Counter()

    old_f1_sum = 0.0
    old_em_sum = 0.0
    new_f1_sum = 0.0
    new_em_sum = 0.0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1

            full_output = rec.get("full_output", "")
            golden_answers = rec.get("golden_answers", [])
            old_extracted = rec.get("extracted_answer", "")
            old_f1 = float(rec.get("f1", 0.0))
            old_em = int(rec.get("em", 0))

            old_f1_sum += old_f1
            old_em_sum += old_em
            if old_extracted in ("[NO_ANSWER]", "[ERROR]", None, ""):
                n_old_no_answer += 1

            new_extracted, source = extract_answer_robust(full_output)
            source_counts[source] += 1

            if new_extracted is None:
                new_f1, new_em = 0.0, 0
                new_extracted_field = "[NO_ANSWER]"
                n_new_no_answer += 1
            else:
                new_f1 = f1_check(new_extracted, golden_answers)
                new_em = em_check(new_extracted, golden_answers)
                new_extracted_field = new_extracted
                if old_extracted in ("[NO_ANSWER]", "[ERROR]", None, ""):
                    n_recovered += 1

            new_f1_sum += new_f1
            new_em_sum += new_em

            if abs(new_f1 - old_f1) > 1e-9 or new_em != old_em:
                n_changed_score += 1

            new_rec = dict(rec)
            new_rec["extracted_answer_old"] = old_extracted
            new_rec["f1_old"] = old_f1
            new_rec["em_old"] = old_em
            new_rec["extracted_answer"] = new_extracted_field
            new_rec["f1"] = new_f1
            new_rec["em"] = new_em
            new_rec["extract_source"] = source
            new_records.append(new_rec)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "input": input_path,
        "output": output_path,
        "num_samples": n_total,
        "old_no_answer": n_old_no_answer,
        "old_no_answer_pct": round(100 * n_old_no_answer / max(1, n_total), 2),
        "new_no_answer": n_new_no_answer,
        "new_no_answer_pct": round(100 * n_new_no_answer / max(1, n_total), 2),
        "recovered_from_no_answer": n_recovered,
        "score_changed": n_changed_score,
        "old_f1": round(old_f1_sum / max(1, n_total), 4),
        "old_em": round(old_em_sum / max(1, n_total), 4),
        "new_f1": round(new_f1_sum / max(1, n_total), 4),
        "new_em": round(new_em_sum / max(1, n_total), 4),
        "delta_f1": round((new_f1_sum - old_f1_sum) / max(1, n_total), 4),
        "delta_em": round((new_em_sum - old_em_sum) / max(1, n_total), 4),
        "extract_source_distribution": dict(source_counts),
    }
    return summary, new_records


def discover_results(root_dir):
    """Find every {nq,hotpot,triviaqa}_results.jsonl under root_dir/outputs/.

    Per-rank shard files (e.g. nq_results_rank0.jsonl) and previously rescored
    files are skipped — only the merged/full results are picked up.
    """
    patterns = [
        "outputs/**/nq_results.jsonl",
        "outputs/**/hotpot_results.jsonl",
        "outputs/**/triviaqa_results.jsonl",
    ]
    found = []
    for pat in patterns:
        for p in glob.glob(os.path.join(root_dir, pat), recursive=True):
            if "_rescored" in os.path.basename(p):
                continue
            if "_rank" in os.path.basename(p):
                continue
            found.append(p)
    return sorted(set(found))


def print_summary(summary):
    print(f"\n=== {summary['input']} ===")
    print(f"  samples:                   {summary['num_samples']}")
    print(f"  NO_ANSWER (old / new):     {summary['old_no_answer']} "
          f"({summary['old_no_answer_pct']}%)  →  {summary['new_no_answer']} "
          f"({summary['new_no_answer_pct']}%)")
    print(f"  recovered from NO_ANSWER:  {summary['recovered_from_no_answer']}")
    print(f"  records w/ changed score:  {summary['score_changed']}")
    print(f"  F1:    {summary['old_f1']:.4f}  →  {summary['new_f1']:.4f} "
          f"(Δ {summary['delta_f1']:+.4f})")
    print(f"  EM:    {summary['old_em']:.4f}  →  {summary['new_em']:.4f} "
          f"(Δ {summary['delta_em']:+.4f})")
    print(f"  extract sources:           {summary['extract_source_distribution']}")
    if summary["output"]:
        print(f"  rescored file:             {summary['output']}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline re-scorer for claim-level eval outputs.",
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a single *_results.jsonl file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write the rescored jsonl. "
                             "Defaults to <input>_rescored.jsonl when --input is set.")
    parser.add_argument("--all", action="store_true",
                        help="Discover and rescore every results file under outputs/.")
    parser.add_argument("--root", type=str,
                        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="Project root for --all discovery (default: repo root).")
    parser.add_argument("--diff", action="store_true",
                        help="Print summary only; do not write a rescored file.")
    parser.add_argument("--summary_json", type=str, default=None,
                        help="If set, write a JSON list of per-file summaries here.")
    args = parser.parse_args()

    targets = []
    if args.all:
        targets = discover_results(args.root)
        if not targets:
            print(f"No results files found under {args.root}/outputs/")
            return
    elif args.input:
        targets = [args.input]
    else:
        parser.error("Pass either --input <file> or --all.")

    summaries = []
    for path in targets:
        if args.diff:
            out_path = None
        elif args.all:
            out_path = path.replace(".jsonl", "_rescored.jsonl")
        else:
            out_path = args.output or path.replace(".jsonl", "_rescored.jsonl")

        summary, _ = rescore_file(path, output_path=out_path, diff=args.diff)
        print_summary(summary)
        summaries.append(summary)

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
