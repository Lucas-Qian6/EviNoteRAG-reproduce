"""
Single-process batch evaluation with claim-level evidence organization
(Pipeline C).

Supports two backends:
  - local: a HuggingFace causal LM loaded on a local GPU (default).
  - gemini: any Gemini-compatible chat completion API exposed through the
            google-genai SDK (e.g. Gemini-3-flash served via gpugeek).

Compare against eval_single_gpu.py (Pipeline B) to measure the effect of
explicit claim-level organization on answer quality.

Usage:
  1. Start the retriever server (BM25 or dense e5).
  2a. Local model:
        python eval_single_gpu_claimlevel.py \
          --model_id /finder/qyj/models/Qwen2.5-7B-Instruct \
          --num_samples 200

  2b. Gemini API (e.g. via gpugeek):
        export GEMINI_API_KEY=...
        python eval_single_gpu_claimlevel.py \
          --backend gemini \
          --gemini_model Vendor2/Gemini-3-flash \
          --gemini_base_url https://api.gpugeek.com \
          --gemini_api_version v1beta \
          --num_samples 200
"""

import argparse
import json
import os
import re
import string
import time

import requests
from datasets import load_dataset
from tqdm import tqdm


# ── Scoring (identical to Pipeline B) ────────────────────────────────────────

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


# ── Claim-level prompt (the key difference from Pipeline B) ──────────────────

def build_prompt(question):
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    return f"""
    ## Background Information  
    # Role Definition  
    You are a specialized **Information Retrieval Agent**. Perform reasoning and use the search tool before providing the final answer.
    You should continue searching until all the required information has been retrieved, and then provide the final answer.

    ## Claim-Level Evidence Organization Rules  
    When retrieving information enclosed in `<information>`, organize its content into atomic claims and the relationships between them, then write the result in a `<summary>` block.

    Steps:
    1. **Decompose**: extract each distinct factual statement as a separate claim, labeled with its source (e.g. `C1 [Doc1]`).
    2. **Filter**: drop claims that are not relevant to the question.
    3. **Relate**: for pairs of relevant claims, mark one of:
       - **corroborates**: claims reinforce each other (same or near-identical fact).
       - **contradicts**: claims conflict or give incompatible information.
       - **complements**: claims address different aspects of the question without overlap.
       - **subsumes**: one claim is strictly more specific than the other on the same point.
    4. **Resolve**: merge corroborating claims into one; on contradiction prefer the more specific claim and note the conflict; keep complementary claims as separate points; on subsumption keep the more specific claim and drop the generic one.

    ## Format Instructions  
    - Use `<search>Your query</search>` to call the search tool.
    - For each `<information>Search result</information>`, provide a structured version in `<summary>`, following the steps above.
    - Only output the final answer inside `<answer></answer>`. The answer should be the specific name, date, number, place, or fact that directly answers the question. Do not include explanations, reasoning, or extra text.
    - Always follow this format strictly.
    - **Answer must be in English. Only English responses will be accepted.**
    - You MUST search for evidence before answering. Do NOT answer based on your own knowledge.
    Note: No searches allowed after answer submission. So avoid answering when uncertain – verify accuracy thoroughly before answering
    Question: {question}
    """


# ── Local-model inference (HuggingFace causal LM) ────────────────────────────

def _local_imports():
    """Lazy import torch/transformers so the gemini backend doesn't need them."""
    import torch
    import transformers
    return torch, transformers


class StopOnSequence:
    """Built lazily to avoid importing transformers at module level."""

    def __init__(self, target_sequences, tokenizer):
        torch, transformers = _local_imports()
        self.torch = torch
        self.target_ids = [
            tokenizer.encode(seq, add_special_tokens=False)
            for seq in target_sequences
        ]
        self.target_lengths = [len(t) for t in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        targets = [self.torch.as_tensor(t, device=input_ids.device) for t in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if self.torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False


def run_inference_local(
    prompt, model, tokenizer, device, stopping_criteria,
    curr_eos, max_turns=4, topk=3, retriever_url="http://127.0.0.1:8000/retrieve",
):
    torch, _ = _local_imports()
    search_template = "\n\n{output_text}<information>{search_results}</information>\n\n"
    turns = 0

    while True:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
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
                    max_new_tokens=1024,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            prompt += output_text
            break

    return prompt, turns


# ── Gemini-API inference (google-genai SDK) ──────────────────────────────────

def make_gemini_client(api_key, base_url, api_version):
    """Build a google-genai client. Imports the SDK lazily."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "The 'google-genai' package is required for --backend gemini. "
            "Install it with: pip install -q -U google-genai"
        ) from exc

    http_kwargs = {}
    if base_url:
        http_kwargs["base_url"] = base_url
    if api_version:
        http_kwargs["api_version"] = api_version

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(**http_kwargs) if http_kwargs else None,
    )
    return client, types


def gemini_generate(client, types_mod, model_id, prompt, max_output_tokens,
                    stop_sequences=None, temperature=0.0,
                    max_retries=3, retry_delay=2.0):
    """Single Gemini call with simple exponential-backoff retry on failure."""
    cfg_kwargs = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }
    if stop_sequences:
        cfg_kwargs["stop_sequences"] = list(stop_sequences)

    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types_mod.GenerateContentConfig(**cfg_kwargs),
            )
            return response.text or ""
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                break
            time.sleep(retry_delay * (2 ** attempt))

    raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {last_exc}")


def run_inference_gemini(
    initial_prompt, client, types_mod, model_id,
    max_turns=4, topk=3, retriever_url="http://127.0.0.1:8000/retrieve",
    max_new_tokens_main=2048, max_new_tokens_final=1024,
):
    """Same agentic loop as the local backend, but driven by Gemini API calls.

    The model is expected to emit `<search>...</search>` to query the retriever
    and `<answer>...</answer>` for the final answer. We pass `</search>` as a
    stop sequence; when the API truncates, we re-append `</search>` so the rest
    of the pipeline (regex extractor) can read it back.
    """
    prompt = initial_prompt
    turns = 0
    search_template = "\n\n{output_text}<information>{search_results}</information>\n\n"

    while True:
        output_text = gemini_generate(
            client, types_mod, model_id, prompt,
            max_output_tokens=max_new_tokens_main,
            stop_sequences=["</search>"],
        )

        last_open = output_text.rfind("<search>")
        last_close = output_text.rfind("</search>")
        last_answer_close = output_text.rfind("</answer>")

        stopped_for_search = (
            last_open > last_close
            and last_open > last_answer_close
        )

        if stopped_for_search:
            output_text = output_text + "</search>"

        if not stopped_for_search:
            prompt += output_text
            break

        tmp_query = get_query(output_text)
        if tmp_query:
            search_results = search(tmp_query, topk=topk, retriever_url=retriever_url)
        else:
            search_results = ""

        prompt += search_template.format(output_text=output_text, search_results=search_results)
        turns += 1

        if turns >= max_turns:
            output_text = gemini_generate(
                client, types_mod, model_id, prompt,
                max_output_tokens=max_new_tokens_final,
            )
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
    parser = argparse.ArgumentParser(description="Claim-level evidence organization eval")
    parser.add_argument("--backend", type=str, default="local",
                        choices=["local", "gemini"],
                        help="Model backend: local HF causal LM or Gemini-compatible API")

    parser.add_argument("--model_id", type=str, default=None,
                        help="HuggingFace model ID or local path (required for --backend local)")

    parser.add_argument("--gemini_model", type=str, default="Vendor2/Gemini-3-flash",
                        help="Model name to send to the Gemini API (e.g. Vendor2/Gemini-3-flash)")
    parser.add_argument("--gemini_api_key", type=str, default=os.environ.get("GEMINI_API_KEY"),
                        help="API key (defaults to $GEMINI_API_KEY)")
    parser.add_argument("--gemini_base_url", type=str, default="https://api.gpugeek.com",
                        help="Override the API base URL (use for OpenAI-compatible gateways like gpugeek)")
    parser.add_argument("--gemini_api_version", type=str, default="v1beta",
                        help="API version for HttpOptions (gpugeek expects v1beta)")

    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of TriviaQA questions (None = full val set)")
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_url", type=str,
                        default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/eval_claimlevel")
    parser.add_argument("--resume_from", type=int, default=0)
    args = parser.parse_args()

    if args.backend == "local" and not args.model_id:
        parser.error("--model_id is required when --backend local")
    if args.backend == "gemini" and not args.gemini_api_key:
        parser.error("--gemini_api_key is required when --backend gemini "
                     "(or set $GEMINI_API_KEY)")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "triviaqa_results.jsonl")

    print("Checking retriever server...")
    try:
        search("test query", topk=1, retriever_url=args.retriever_url)
        print("  Retriever is up.")
    except Exception as e:
        print(f"  ERROR: Cannot reach retriever at {args.retriever_url}")
        print(f"  {e}")
        return

    label = f"{args.num_samples} samples" if args.num_samples else "full val set"
    print(f"Loading TriviaQA validation data ({label})...")
    data = load_triviaqa(args.num_samples)
    print(f"  Loaded {len(data)} questions")

    if args.backend == "local":
        torch, transformers = _local_imports()
        print(f"Loading local model: {args.model_id}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        print(f"  Model loaded on {device}")

        curr_eos = [151645, 151643]
        target_sequences = [
            "</search>", " </search>",
            "</search>\n", " </search>\n",
            "</search>\n\n", " </search>\n\n",
        ]
        stopping_criteria = transformers.StoppingCriteriaList(
            [StopOnSequence(target_sequences, tokenizer)]
        )
        gemini_client = None
        gemini_types = None
        model_label = args.model_id
    else:
        print(f"Initialising Gemini client for {args.gemini_model} "
              f"(base_url={args.gemini_base_url}, api_version={args.gemini_api_version})")
        gemini_client, gemini_types = make_gemini_client(
            api_key=args.gemini_api_key,
            base_url=args.gemini_base_url,
            api_version=args.gemini_api_version,
        )
        torch = None
        tokenizer = None
        model = None
        device = None
        stopping_criteria = None
        curr_eos = None
        model_label = args.gemini_model

    f1_scores = []
    em_scores = []
    start_time = time.time()

    for idx in tqdm(range(args.resume_from, len(data)), desc="Evaluating"):
        item = data[idx]
        question = item["question"]
        golden_answers = item["golden_answers"]
        raw_prompt = build_prompt(question)

        try:
            if args.backend == "local":
                if tokenizer.chat_template:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": raw_prompt}],
                        add_generation_prompt=True, tokenize=False,
                    )
                else:
                    prompt = raw_prompt
                full_output, num_turns = run_inference_local(
                    prompt, model, tokenizer, device, stopping_criteria,
                    curr_eos,
                    max_turns=args.max_turns,
                    topk=args.topk,
                    retriever_url=args.retriever_url,
                )
            else:
                full_output, num_turns = run_inference_gemini(
                    raw_prompt, gemini_client, gemini_types, args.gemini_model,
                    max_turns=args.max_turns,
                    topk=args.topk,
                    retriever_url=args.retriever_url,
                    max_new_tokens_main=2048,
                    max_new_tokens_final=1024,
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
            "backend": args.backend,
            "model": model_label,
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if torch is not None and torch.cuda.is_available():
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

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0

    summary = {
        "dataset": "triviaqa",
        "pipeline": "claim-level",
        "num_samples": len(f1_scores),
        "avg_f1": round(avg_f1, 4),
        "avg_em": round(avg_em, 4),
        "backend": args.backend,
        "model": model_label,
        "max_turns": args.max_turns,
        "topk": args.topk,
    }
    summary_file = os.path.join(args.output_dir, "triviaqa_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS — Claim-Level Pipeline ({len(f1_scores)} questions, backend={args.backend})")
    print(f"  F1:  {avg_f1:.4f}")
    print(f"  EM:  {avg_em:.4f}")
    print(f"  Results:  {output_file}")
    print(f"  Summary:  {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
