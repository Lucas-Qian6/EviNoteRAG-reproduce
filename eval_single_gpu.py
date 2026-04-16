"""
Single-GPU batch evaluation for EviNote-RAG on TriviaQA.
Adapted from infer.py to loop over a dataset and compute F1/EM scores.

Usage:
  1. Start the BM25 retriever server in a separate terminal (with `retriever` conda env):
       python RAG/search/retrieval_server.py \
         --retriever_name bm25 \
         --index_path /root/finder/qyj/sparse_retriever/bm25 \
         --corpus_path /root/finder/qyj/sparse_retriever/wiki-18.jsonl \
         --topk 3

  2. Run evaluation (with `EviNoteRAG` conda env):
       python eval_single_gpu.py --model_id dayll/EviNoteRAG-7B --num_samples 200
"""

import argparse
import json
import os
import re
import string
import time

import torch
import transformers
import requests
from datasets import load_dataset
from tqdm import tqdm


# ── Scoring ──────────────────────────────────────────────────────────────────

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


# ── Answer / query extraction ───────────────────────────────────────────────

def extract_answer(text):
    """Extract the last <answer>...</answer> from the full conversation.
    Returns None if the model didn't produce an answer (only the instruction
    template mention counts as 1 match, which we skip)."""
    matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    if len(matches) <= 1:
        return None
    answer = matches[-1].group(1).strip()
    return answer if answer else None


def get_query(text):
    matches = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    return matches[-1] if matches else None


# ── Retriever client ─────────────────────────────────────────────────────────

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
        parts.append(f"Doc {idx+1}(Title: {title}) {text}")
    return "\n".join(parts)


# ── Prompt builder (same template as infer.py) ───────────────────────────────

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
    - Only output the final answer inside `<answer></answer>`. Do not include explanations, reasoning, or extra text.
    - If it's a yes/no question, respond only with `yes` or `no`.
    - Always follow this format strictly.
    - **Answer must be in English. Only English responses will be accepted.**
    Note: No searches allowed after answer submission. So avoid answering when uncertain – verify accuracy thoroughly before answering
    Question: {question}
    """


# ── Stopping criteria (same as infer.py) ────────────────────────────────────

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


# ── Agentic inference loop ───────────────────────────────────────────────────

def run_inference(
    prompt, model, tokenizer, device, stopping_criteria,
    curr_eos, max_turns=4, topk=3, retriever_url="http://127.0.0.1:8000/retrieve",
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

        prompt += search_template.format(output_text=output_text, search_results=search_results)
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


# ── Data loading ─────────────────────────────────────────────────────────────

def load_triviaqa(num_samples=None):
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    data = []
    for item in ds:
        data.append({
            "question": item["question"],
            "golden_answers": item["answer"]["aliases"],
        })
    return data


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Single-GPU EviNote-RAG evaluation")
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of TriviaQA questions to evaluate")
    parser.add_argument("--max_turns", type=int, default=4,
                        help="Max search turns per question")
    parser.add_argument("--topk", type=int, default=3,
                        help="Number of docs per retrieval call")
    parser.add_argument("--retriever_url", type=str,
                        default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/eval_single_gpu")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume from this question index (0-based)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "triviaqa_results.jsonl")

    # ── Retriever health check ───────────────────────────────────────────
    print("Checking retriever server...")
    try:
        test = search("test query", topk=1, retriever_url=args.retriever_url)
        print("  Retriever is up.")
    except Exception as e:
        print(f"  ERROR: Cannot reach retriever at {args.retriever_url}")
        print(f"  {e}")
        print("  Start the retriever server first (see eval_single_gpu.sh).")
        return

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading TriviaQA validation data ({args.num_samples} samples)...")
    data = load_triviaqa(args.num_samples)
    print(f"  Loaded {len(data)} questions")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model: {args.model_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"  Model loaded on {device}")

    curr_eos = [151645, 151643]  # Qwen2.5 EOS tokens
    target_sequences = [
        "</search>", " </search>",
        "</search>\n", " </search>\n",
        "</search>\n\n", " </search>\n\n",
    ]
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)]
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    f1_scores = []
    em_scores = []
    start_time = time.time()

    for idx in tqdm(range(args.resume_from, len(data)), desc="Evaluating"):
        item = data[idx]
        question = item["question"]
        golden_answers = item["golden_answers"]

        raw_prompt = build_prompt(question)
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                add_generation_prompt=True, tokenize=False,
            )
        else:
            prompt = raw_prompt

        try:
            full_output, num_turns = run_inference(
                prompt, model, tokenizer, device, stopping_criteria,
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

        except Exception as e:
            print(f"\n[ERROR] Question {idx}: {e}")
            full_output = f"[ERROR] {e}"
            extracted_answer = "[ERROR]"
            f1, em = 0.0, 0
            num_turns = -1

        f1_scores.append(f1)
        em_scores.append(em)

        result = {
            "index": idx,
            "question": question,
            "golden_answers": golden_answers,
            "extracted_answer": extracted_answer,
            "f1": f1,
            "em": em,
            "num_turns": num_turns,
            "full_output": full_output,
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_em = sum(em_scores) / len(em_scores)
            elapsed = time.time() - start_time
            per_q = elapsed / len(f1_scores)
            remaining = per_q * (len(data) - args.resume_from - len(f1_scores))
            print(
                f"\n  [{idx+1}/{len(data)}] "
                f"F1={avg_f1:.4f}  EM={avg_em:.4f}  "
                f"({per_q:.1f}s/q, ~{remaining/60:.0f}min left)"
            )

    # ── Summary ──────────────────────────────────────────────────────────
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0

    summary = {
        "dataset": "triviaqa",
        "num_samples": len(f1_scores),
        "avg_f1": round(avg_f1, 4),
        "avg_em": round(avg_em, 4),
        "model": args.model_id,
        "max_turns": args.max_turns,
        "topk": args.topk,
    }
    summary_file = os.path.join(args.output_dir, "triviaqa_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS  ({len(f1_scores)} questions)")
    print(f"  F1:  {avg_f1:.4f}")
    print(f"  EM:  {avg_em:.4f}")
    print(f"  Results:  {output_file}")
    print(f"  Summary:  {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
