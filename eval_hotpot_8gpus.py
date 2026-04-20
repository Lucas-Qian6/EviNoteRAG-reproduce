"""
Multi-GPU batch evaluation for EviNote-RAG on HotpotQA.

Mirrors `eval_nq_8gpus.py` but loads HotpotQA from Hugging Face. The retrieval
and generation loop, the prompt, the scoring, and the multi-GPU sharding are
all identical to the NQ evidence-level script.

Usage:
  1. Start the BM25 retriever server in a separate terminal (with `retriever`
     conda env):
       python RAG/search/retrieval_server.py \
         --retriever_name bm25 \
         --index_path /path/to/bm25 \
         --corpus_path /path/to/wiki-18.jsonl \
         --topk 3

  2. Run evaluation (with `EviNoteRAG` conda env):
       export HF_ENDPOINT=https://hf-mirror.com
       python eval_hotpot_8gpus.py \
         --model_id dayll/EviNoteRAG-7B \
         --num_gpus 8 \
         --config distractor
"""

import argparse
import json
import os
import re
import string
import time
from glob import glob

import requests
import torch
import torch.multiprocessing as mp
import transformers
from datasets import load_dataset
from tqdm import tqdm


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


def extract_answer(text):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if len(matches) <= 1:
        return None
    answer = matches[-1].group(1).strip()
    return answer if answer else None


def get_query(text):
    matches = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    return matches[-1] if matches else None


def search(query, topk=3, retriever_url="http://127.0.0.1:8000/retrieve"):
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    resp = requests.post(retriever_url, json=payload, timeout=30)
    resp.raise_for_status()
    results = resp.json()["result"]

    parts = []
    for idx, doc_item in enumerate(results[0]):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        parts.append(f"Doc {idx + 1}(Title: {title}) {text}")
    return "\n".join(parts)


def build_prompt(question):
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    return f"""
    ## Background Information
    # Role Definition
    You are a specialized **Information Retrieval Agent**. Perform reasoning and use the search tool before providing the final answer.
    You should continue searching until all the required information has been retrieved, and then provide the final answer.

    ## Note-Taking Rules
    When retrieving information enclosed in `<information>`, summarize its content and use the following markers to highlight key or uncertain elements:

    There are two types of markers:
    1. `-` (Uncertainty): Marks ambiguous or uncertain information.
    Example: `-He picked up Jenny-` (Uncertain who "he" refers to).
    2. `*` (Key Info): Highlights important or critical details.
    Example: *Built in 1900* (The year is essential).

    ## Format Instructions
    - Use `<search>Your query</search>` to call the search tool.
    - For each `<information>Search result</information>`, provide a summarized version in `<summary>`, using the above markers to indicate key or uncertain information.
    - Only output the final answer inside `<answer></answer>`. The answer should be the specific name, date, number, place, or fact that directly answers the question. Do not include explanations, reasoning, or extra text.
    - Always follow this format strictly.
    - **Answer must be in English. Only English responses will be accepted.**
    - You MUST search for evidence before answering. Do NOT answer based on your own knowledge.
    Note: No searches allowed after answer submission. So avoid answering when uncertain - verify accuracy thoroughly before answering.
    Question: {question}
    """


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(seq, add_special_tokens=False)
            for seq in target_sequences
        ]
        self.target_lengths = [len(t) for t in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(t, device=input_ids.device) for t in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False

        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False


def run_inference(
    prompt,
    model,
    tokenizer,
    device,
    stopping_criteria,
    curr_eos,
    max_turns=4,
    topk=3,
    retriever_url="http://127.0.0.1:8000/retrieve",
):
    search_template = "\n\n{output_text}<information>{search_results}</information>\n\n"
    turns = 0

    while True:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if outputs[0][-1].item() in curr_eos:
            prompt += output_text
            break

        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            search_results = search(tmp_query, topk=topk, retriever_url=retriever_url)
        else:
            search_results = ""

        prompt += search_template.format(
            output_text=output_text,
            search_results=search_results,
        )
        turns += 1

        if turns >= max_turns:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            prompt += output_text
            break

    return prompt, turns


def load_hotpotqa(num_samples=None, dataset_id=None, split="validation",
                  config="distractor"):
    """Load HotpotQA from Hugging Face.

    HotpotQA on HF requires a config: "distractor" (closed-set, the standard
    benchmark setting) or "fullwiki" (open-domain). For RAG eval against a
    Wikipedia retriever the config choice does NOT change the questions or
    gold answers - only the bundled context fields, which we ignore.
    """
    dataset_ids = [dataset_id] if dataset_id else [
        "hotpotqa/hotpot_qa",
        "hotpot_qa",
    ]

    last_error = None
    ds = None
    used_dataset_id = None
    for candidate in dataset_ids:
        try:
            ds = load_dataset(candidate, config, split=split,
                              trust_remote_code=True)
            used_dataset_id = candidate
            break
        except Exception as exc:
            last_error = exc

    if ds is None:
        raise RuntimeError(
            f"Failed to load HotpotQA from {dataset_ids} (config={config}, "
            f"split={split}): {last_error}"
        )

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    data = []
    for idx, item in enumerate(ds):
        answer = item.get("answer")
        if answer is None or not str(answer).strip():
            raise ValueError(f"No valid answer found for HotpotQA example {idx}")

        data.append(
            {
                "index": idx,
                "question": item["question"],
                "golden_answers": [str(answer).strip()],
            }
        )

    return data, used_dataset_id


def rank_output_path(output_dir, rank):
    return os.path.join(output_dir, f"hotpot_results_rank{rank}.jsonl")


def rank_summary_path(output_dir, rank):
    return os.path.join(output_dir, f"hotpot_summary_rank{rank}.json")


def load_existing_records(path):
    records = {}
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" in record:
                records[record["index"]] = record

    return records


def run_worker(rank, args, data):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    shard = data[rank::args.num_gpus]
    shard_indices = [item["index"] for item in shard]
    output_file = rank_output_path(args.output_dir, rank)
    summary_file = rank_summary_path(args.output_dir, rank)

    existing_records = load_existing_records(output_file)
    completed_records = [existing_records[idx] for idx in shard_indices if idx in existing_records]
    pending_shard = [item for item in shard if item["index"] not in existing_records]

    if not shard or not pending_shard:
        rank_summary = {
            "dataset": "hotpot_qa",
            "config": args.config,
            "dataset_id": args.dataset_id_resolved,
            "split": args.split,
            "model": args.model_id,
            "gpu_rank": rank,
            "num_questions": len(shard),
            "completed_questions": len(completed_records),
            "avg_f1": (
                round(sum(record["f1"] for record in completed_records) / len(completed_records), 4)
                if completed_records
                else 0.0
            ),
            "avg_em": (
                round(sum(record["em"] for record in completed_records) / len(completed_records), 4)
                if completed_records
                else 0.0
            ),
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(rank_summary, f, indent=2)
        print(
            f"[GPU {rank}] No pending HotpotQA questions assigned. "
            f"Questions in shard: {len(shard)}."
        )
        return

    print(
        f"[GPU {rank}] Loading model {args.model_id} on {device} "
        f"for {len(shard)} HotpotQA questions ({len(pending_shard)} pending)."
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{rank}"},
        low_cpu_mem_usage=True,
    )
    model.eval()

    curr_eos = [151645, 151643]
    target_sequences = [
        "</search>",
        " </search>",
        "</search>\n",
        " </search>\n",
        "</search>\n\n",
        " </search>\n\n",
    ]
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)]
    )

    f1_scores = [record["f1"] for record in completed_records]
    em_scores = [record["em"] for record in completed_records]
    start_time = time.time()

    progress = tqdm(
        pending_shard,
        total=len(shard),
        initial=len(completed_records),
        desc=f"GPU {rank}",
        position=rank,
        leave=True,
    )

    for item in progress:
        question = item["question"]
        golden_answers = item["golden_answers"]

        raw_prompt = build_prompt(question)
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt = raw_prompt

        try:
            full_output, num_turns = run_inference(
                prompt,
                model,
                tokenizer,
                device,
                stopping_criteria,
                curr_eos,
                max_turns=args.max_turns,
                topk=args.topk,
                retriever_url=args.retriever_url,
            )

            extracted_answer = extract_answer(full_output)
            if extracted_answer:
                f1 = f1_check(extracted_answer, golden_answers)
                em = em_check(extracted_answer, golden_answers)
            else:
                f1, em = 0.0, 0
                extracted_answer = "[NO_ANSWER]"

        except Exception as exc:
            print(f"\n[GPU {rank}] ERROR on question {item['index']}: {exc}")
            full_output = f"[ERROR] {exc}"
            extracted_answer = "[ERROR]"
            f1, em = 0.0, 0
            num_turns = -1

        f1_scores.append(f1)
        em_scores.append(em)

        result = {
            "index": item["index"],
            "question": question,
            "golden_answers": golden_answers,
            "extracted_answer": extracted_answer,
            "f1": f1,
            "em": em,
            "num_turns": num_turns,
            "full_output": full_output,
            "gpu_rank": rank,
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        done = len(f1_scores)
        if done:
            avg_f1 = sum(f1_scores) / done
            avg_em = sum(em_scores) / done
            elapsed = time.time() - start_time
            processed_now = max(1, done - len(completed_records))
            per_q = elapsed / processed_now
            remaining = per_q * (len(shard) - done)
            progress.set_postfix(
                avg_f1=f"{avg_f1:.4f}",
                avg_em=f"{avg_em:.4f}",
                eta_min=f"{remaining / 60:.1f}",
            )

    rank_summary = {
        "dataset": "hotpot_qa",
        "config": args.config,
        "dataset_id": args.dataset_id_resolved,
        "split": args.split,
        "model": args.model_id,
        "gpu_rank": rank,
        "num_questions": len(shard),
        "completed_questions": len(f1_scores),
        "avg_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "avg_em": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(rank_summary, f, indent=2)


def merge_results(output_dir, model_id, num_gpus, dataset_id, split, max_turns,
                  topk, config):
    merged = {}
    for path in sorted(glob(os.path.join(output_dir, "hotpot_results_rank*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                merged[record["index"]] = record

    merged_records = [merged[idx] for idx in sorted(merged)]
    merged_output_file = os.path.join(output_dir, "hotpot_results.jsonl")
    with open(merged_output_file, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    f1_scores = [record["f1"] for record in merged_records]
    em_scores = [record["em"] for record in merged_records]
    summary = {
        "dataset": "hotpot_qa",
        "config": config,
        "dataset_id": dataset_id,
        "split": split,
        "num_samples": len(merged_records),
        "avg_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "avg_em": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
        "model": model_id,
        "num_gpus": num_gpus,
        "max_turns": max_turns,
        "topk": topk,
    }
    summary_file = os.path.join(output_dir, "hotpot_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return merged_output_file, summary_file, summary


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU EviNote-RAG evaluation on HotpotQA"
    )
    parser.add_argument("--model_id", type=str, required=True, help="HF model ID or local path")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of HotpotQA validation questions")
    parser.add_argument("--max_turns", type=int, default=4, help="Max search turns per question")
    parser.add_argument("--topk", type=int, default=3, help="Number of docs per retrieval call")
    parser.add_argument(
        "--retriever_url",
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="Retriever server endpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/eval_hotpot_8gpus",
        help="Directory for per-rank and merged outputs",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs / worker processes to launch",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help="Optional Hugging Face dataset ID override for HotpotQA",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="distractor",
        choices=["distractor", "fullwiki"],
        help="HotpotQA config (distractor is the standard benchmark setting)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no GPU was detected.")

    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        raise ValueError(
            f"Requested {args.num_gpus} GPUs, but only {available_gpus} are available."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print("Checking retriever server...")
    search("test query", topk=1, retriever_url=args.retriever_url)
    print("  Retriever is up.")

    label = f"{args.num_samples} samples" if args.num_samples else "full validation set"
    print(f"Loading HotpotQA ({label}, config={args.config})...")
    data, dataset_id = load_hotpotqa(
        num_samples=args.num_samples,
        dataset_id=args.dataset_id,
        split=args.split,
        config=args.config,
    )
    args.dataset_id_resolved = dataset_id
    print(f"  Loaded {len(data)} questions from {dataset_id}")

    mp.spawn(
        run_worker,
        args=(args, data),
        nprocs=args.num_gpus,
        join=True,
    )

    merged_output_file, summary_file, summary = merge_results(
        output_dir=args.output_dir,
        model_id=args.model_id,
        num_gpus=args.num_gpus,
        dataset_id=dataset_id,
        split=args.split,
        max_turns=args.max_turns,
        topk=args.topk,
        config=args.config,
    )

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({summary['num_samples']} questions)")
    print(f"  F1:  {summary['avg_f1']:.4f}")
    print(f"  EM:  {summary['avg_em']:.4f}")
    print(f"  Results:  {merged_output_file}")
    print(f"  Summary:  {summary_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
