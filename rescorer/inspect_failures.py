"""
Helper for eyeballing failure cases after re-scoring.

Examples:

    # Show 5 cases where the new extractor recovered the answer
    python rescorer/inspect_failures.py \
        --input outputs/eval_nq_8gpus_claimlevel_evinote/nq_results_rescored.jsonl \
        --mode recovered --n 5

    # Show 5 cases that are still NO_ANSWER after rescoring
    python rescorer/inspect_failures.py \
        --input outputs/eval_nq_8gpus_claimlevel/nq_results_rescored.jsonl \
        --mode still_no_answer --n 5

    # Show 5 cases where the new score went down (should be rare)
    python rescorer/inspect_failures.py \
        --input outputs/eval_nq_8gpus_claimlevel_evinote/nq_results_rescored.jsonl \
        --mode regressions --n 5
"""

import argparse
import json


def iter_records(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def matches_mode(rec, mode):
    old = rec.get("extracted_answer_old", rec.get("extracted_answer"))
    new = rec.get("extracted_answer", "[NO_ANSWER]")
    f1_old = float(rec.get("f1_old", 0.0))
    f1_new = float(rec.get("f1", 0.0))

    if mode == "recovered":
        return old in ("[NO_ANSWER]", "[ERROR]") and new not in ("[NO_ANSWER]", "[ERROR]")
    if mode == "still_no_answer":
        return new in ("[NO_ANSWER]", "[ERROR]")
    if mode == "regressions":
        return f1_new < f1_old - 1e-9
    if mode == "improvements":
        return f1_new > f1_old + 1e-9
    if mode == "all":
        return True
    raise ValueError(f"Unknown mode: {mode}")


def show(rec, tail_chars=600):
    full = rec.get("full_output", "")
    print(f"Q:       {rec.get('question')}")
    print(f"GOLD:    {rec.get('golden_answers')}")
    print(f"TURNS:   {rec.get('num_turns')}")
    print(f"OLD ans: {rec.get('extracted_answer_old')}  "
          f"(f1={rec.get('f1_old')}, em={rec.get('em_old')})")
    print(f"NEW ans: {rec.get('extracted_answer')}  "
          f"(f1={rec.get('f1')}, em={rec.get('em')}, src={rec.get('extract_source')})")
    print(f"TAIL:    ...{full[-tail_chars:]}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to a *_rescored.jsonl produced by rescore.py.")
    parser.add_argument(
        "--mode",
        choices=["recovered", "still_no_answer", "regressions", "improvements", "all"],
        default="recovered",
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--tail_chars", type=int, default=600)
    args = parser.parse_args()

    shown = 0
    for rec in iter_records(args.input):
        if matches_mode(rec, args.mode):
            show(rec, tail_chars=args.tail_chars)
            shown += 1
            if shown >= args.n:
                break

    if shown == 0:
        print(f"No records match mode={args.mode} in {args.input}")


if __name__ == "__main__":
    main()
