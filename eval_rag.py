"""
Unified multi-worker EviNote-RAG evaluation script.

Replaces the six per-dataset 8-GPU files with one entry point selectable by
flags:

  --dataset   {nq, hotpot, trivia}
  --pipeline  {note, claim}              (note = original note-taking, claim = claim-level)
  --backend   {local, gemini}            (local HF causal LM, or Gemini-compatible API)

The retrieval-and-generation loop, scoring, sharding, and output schema are
identical across all combinations - only the dataset loader, prompt template,
and (for the API backend) the generation call differ.

Examples:

  # NQ, claim-level, local 8-GPU evaluation
  python eval_rag.py --dataset nq --pipeline claim \\
      --backend local --model_id dayll/EviNoteRAG-7B --num_workers 8

  # HotpotQA, note-taking, Gemini-3-flash via gpugeek (no GPU needed)
  export GEMINI_API_KEY=...
  python eval_rag.py --dataset hotpot --pipeline note \\
      --backend gemini --gemini_model Vendor2/Gemini-3-flash \\
      --num_workers 4 --config distractor

  # TriviaQA, claim-level, local 8-GPU
  python eval_rag.py --dataset trivia --pipeline claim \\
      --backend local --model_id dayll/EviNoteRAG-7B --num_workers 8
"""

import argparse
import json
import os
import re
import string
import time
from glob import glob

import requests
from datasets import load_dataset
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────────
# Scoring
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# Answer / query extraction
# ────────────────────────────────────────────────────────────────────────────

def extract_answer(text):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if len(matches) <= 1:
        return None
    answer = matches[-1].group(1).strip()
    return answer if answer else None


def get_query(text):
    matches = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    return matches[-1] if matches else None


# ────────────────────────────────────────────────────────────────────────────
# Retriever client
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ────────────────────────────────────────────────────────────────────────────

def build_prompt_note(question):
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


def build_prompt_claim(question):
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


PIPELINES = {
    "note": {
        "build_prompt": build_prompt_note,
        "max_new_tokens_main": 1024,
        "max_new_tokens_final": 512,
    },
    "claim": {
        "build_prompt": build_prompt_claim,
        "max_new_tokens_main": 10240,
        "max_new_tokens_final": 5120,
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ────────────────────────────────────────────────────────────────────────────

def _normalize_trivia_answers(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, dict):
        if raw.get("aliases"):
            candidates = list(raw["aliases"])
            if raw.get("value"):
                candidates.append(raw["value"])
        elif raw.get("normalized_aliases"):
            candidates = list(raw["normalized_aliases"])
        elif "text" in raw:
            candidates = raw["text"]
        elif raw.get("value"):
            candidates = [raw["value"]]
        else:
            candidates = list(raw.values())
    else:
        candidates = list(raw)

    out, seen = [], set()
    for ans in candidates:
        if ans is None:
            continue
        if isinstance(ans, dict):
            ans = ans.get("text") or ans.get("value") or ""
        ans = str(ans).strip()
        if ans and ans not in seen:
            seen.add(ans)
            out.append(ans)
    return out


def _normalize_nq_answers(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, dict):
        if "text" in raw:
            candidates = raw["text"]
        elif "aliases" in raw:
            candidates = raw["aliases"]
        else:
            candidates = list(raw.values())
    else:
        candidates = list(raw)
    out = []
    for ans in candidates:
        if ans is None:
            continue
        if isinstance(ans, dict):
            ans = ans.get("text", "")
        ans = str(ans).strip()
        if ans:
            out.append(ans)
    return out


def load_nq(num_samples, dataset_id, config, split):
    candidates = [dataset_id] if dataset_id else [
        "google-research-datasets/nq_open",
        "nq_open",
    ]
    last_error, ds, used = None, None, None
    for cand in candidates:
        try:
            ds = load_dataset(cand, split=split)
            used = cand
            break
        except Exception as exc:
            last_error = exc
    if ds is None:
        raise RuntimeError(f"Failed to load NQ Open from {candidates}: {last_error}")

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    data = []
    for idx, item in enumerate(ds):
        answers = _normalize_nq_answers(item.get("answer") or item.get("answers"))
        if not answers:
            raise ValueError(f"No valid answers found for NQ example {idx}")
        data.append({
            "index": idx,
            "question": item["question"],
            "golden_answers": answers,
        })
    return data, used


def load_hotpot(num_samples, dataset_id, config, split):
    candidates = [dataset_id] if dataset_id else [
        "hotpotqa/hotpot_qa",
        "hotpot_qa",
    ]
    cfg = config or "distractor"
    last_error, ds, used = None, None, None
    for cand in candidates:
        try:
            ds = load_dataset(cand, cfg, split=split, trust_remote_code=True)
            used = cand
            break
        except Exception as exc:
            last_error = exc
    if ds is None:
        raise RuntimeError(
            f"Failed to load HotpotQA from {candidates} (config={cfg}, split={split}): {last_error}"
        )

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    data = []
    for idx, item in enumerate(ds):
        answer = item.get("answer")
        if answer is None or not str(answer).strip():
            raise ValueError(f"No valid answer for HotpotQA example {idx}")
        data.append({
            "index": idx,
            "question": item["question"],
            "golden_answers": [str(answer).strip()],
        })
    return data, used


def load_trivia(num_samples, dataset_id, config, split):
    candidates = [dataset_id] if dataset_id else [
        "mandarjoshi/trivia_qa",
        "trivia_qa",
    ]
    cfg = config or "rc.nocontext"
    last_error, ds, used = None, None, None
    for cand in candidates:
        try:
            ds = load_dataset(cand, cfg, split=split)
            used = cand
            break
        except Exception as exc:
            last_error = exc
    if ds is None:
        raise RuntimeError(
            f"Failed to load TriviaQA from {candidates} (config={cfg}): {last_error}"
        )

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    data = []
    for idx, item in enumerate(ds):
        answers = _normalize_trivia_answers(item.get("answer") or item.get("answers"))
        if not answers:
            raise ValueError(f"No valid answers for TriviaQA example {idx}")
        data.append({
            "index": idx,
            "question": item["question"],
            "golden_answers": answers,
        })
    return data, used

def load_parquet(num_samples, dataset_id, config, split):
    """Load questions directly from m_test.parquet so we evaluate on the
    same 500 questions per benchmark that EviNote / Search-R1 use.

    Args:
        dataset_id: path to the parquet file (e.g. ./data_preprocess/data/m_test.parquet)
        config: data_source prefix to filter on, e.g. 'triviaqa', 'nq', 'hotpotqa',
                'popqa', '2wikimultihopqa', 'musique', 'bamboogle'.
                If None, loads all rows.
        split: ignored.
    """
    import pandas as pd

    parquet_path = dataset_id or "./data_preprocess/data/m_test.parquet"
    df = pd.read_parquet(parquet_path)

    if config:
        # match both "<name>" and "<name>_val"
        mask = df["data_source"].astype(str).str.startswith(config)
        df = df[mask].reset_index(drop=True)

    if num_samples and num_samples < len(df):
        df = df.iloc[:num_samples].reset_index(drop=True)

    def extract_q(prompt_content: str) -> str:
        if "Question:" not in prompt_content:
            return prompt_content.strip()
        return prompt_content.split("Question:")[-1].strip().splitlines()[0].strip()

    data = []
    for idx, row in df.iterrows():
        q = extract_q(row["prompt"][0]["content"])
        golds = list(row["reward_model"]["ground_truth"]["target"])
        if not golds:
            raise ValueError(f"No gold answers for parquet row {idx}")
        data.append({
            "index": int(idx),
            "question": q,
            "golden_answers": golds,
        })
    return data, parquet_path

DATASETS = {
    "nq": {
        "loader": load_nq,
        "prefix": "nq",
        "default_config": None,
        "config_help": "NQ Open ignores --config",
    },
    "hotpot": {
        "loader": load_hotpot,
        "prefix": "hotpot",
        "default_config": "distractor",
        "config_help": "HotpotQA: distractor or fullwiki",
    },
    "trivia": {
        "loader": load_trivia,
        "prefix": "trivia",
        "default_config": "rc.nocontext",
        "config_help": "TriviaQA: rc, rc.nocontext, unfiltered, unfiltered.nocontext",
    },
    "parquet": {
        "loader": load_parquet,
        "prefix": "parquet",
        "default_config": None,
        "config_help": (
            "Use --config <data_source_prefix> to pick a subset, e.g. "
            "triviaqa, nq, hotpotqa, popqa, 2wikimultihopqa, musique, bamboogle. "
            "Use --dataset_id to point at a different parquet path."
        ),
    }, 
}


# ────────────────────────────────────────────────────────────────────────────
# Local backend (HuggingFace causal LM)
# ────────────────────────────────────────────────────────────────────────────

def _local_imports():
    import torch
    import transformers
    return torch, transformers


def make_stop_on_sequence_class(torch_mod, transformers_mod):
    class StopOnSequence(transformers_mod.StoppingCriteria):
        def __init__(self, target_sequences, tokenizer):
            self.target_ids = [
                tokenizer.encode(seq, add_special_tokens=False)
                for seq in target_sequences
            ]
            self.target_lengths = [len(t) for t in self.target_ids]

        def __call__(self, input_ids, scores, **kwargs):
            targets = [torch_mod.as_tensor(t, device=input_ids.device) for t in self.target_ids]
            if input_ids.shape[1] < min(self.target_lengths):
                return False
            for i, target in enumerate(targets):
                if torch_mod.equal(input_ids[0, -self.target_lengths[i]:], target):
                    return True
            return False
    return StopOnSequence


def run_inference_local(
    prompt, model, tokenizer, device, stopping_criteria, curr_eos,
    max_turns, topk, retriever_url,
    max_new_tokens_main, max_new_tokens_final,
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
                max_new_tokens=max_new_tokens_main,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=curr_eos,
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
                    max_new_tokens=max_new_tokens_final,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=curr_eos,
                    do_sample=False,
                )
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            prompt += output_text
            break

    return prompt, turns


# ────────────────────────────────────────────────────────────────────────────
# Gemini backend (google-genai SDK)
# ────────────────────────────────────────────────────────────────────────────

def make_gemini_client(api_key, base_url, api_version):
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
    max_turns, topk, retriever_url,
    max_new_tokens_main, max_new_tokens_final,
):
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


# ────────────────────────────────────────────────────────────────────────────
# Per-rank shard runner (used by both backends, spawned via mp.Process)
# ────────────────────────────────────────────────────────────────────────────

def rank_output_path(output_dir, prefix, rank):
    return os.path.join(output_dir, f"{prefix}_results_rank{rank}.jsonl")


def rank_summary_path(output_dir, prefix, rank):
    return os.path.join(output_dir, f"{prefix}_summary_rank{rank}.json")


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

    prefix = DATASETS[args.dataset]["prefix"]
    pipeline_cfg = PIPELINES[args.pipeline]
    build_prompt = pipeline_cfg["build_prompt"]
    max_main = pipeline_cfg["max_new_tokens_main"]
    max_final = pipeline_cfg["max_new_tokens_final"]

    shard = data[rank::args.num_workers]
    shard_indices = [item["index"] for item in shard]
    output_file = rank_output_path(args.output_dir, prefix, rank)
    summary_file = rank_summary_path(args.output_dir, prefix, rank)

    existing_records = load_existing_records(output_file)
    completed_records = [existing_records[idx] for idx in shard_indices if idx in existing_records]
    pending_shard = [item for item in shard if item["index"] not in existing_records]

    base_summary_meta = {
        "dataset": args.dataset,
        "dataset_id": args.dataset_id_resolved,
        "config": args.config,
        "split": args.split,
        "pipeline": args.pipeline,
        "backend": args.backend,
        "model": args.model_label,
        "worker_rank": rank,
        "num_workers": args.num_workers,
        "num_questions": len(shard),
    }

    if not shard or not pending_shard:
        rank_summary = {
            **base_summary_meta,
            "completed_questions": len(completed_records),
            "avg_f1": (
                round(sum(r["f1"] for r in completed_records) / len(completed_records), 4)
                if completed_records else 0.0
            ),
            "avg_em": (
                round(sum(r["em"] for r in completed_records) / len(completed_records), 4)
                if completed_records else 0.0
            ),
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(rank_summary, f, indent=2)
        print(f"[worker {rank}] Nothing to do. shard={len(shard)} already_done={len(completed_records)}")
        return

    if args.backend == "local":
        torch, transformers = _local_imports()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(
            f"[worker {rank}] Loading local model {args.model_id} on {device} "
            f"for {len(shard)} {args.dataset} questions ({len(pending_shard)} pending)."
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
            "</search>", " </search>",
            "</search>\n", " </search>\n",
            "</search>\n\n", " </search>\n\n",
        ]
        StopOnSequence = make_stop_on_sequence_class(torch, transformers)
        stopping_criteria = transformers.StoppingCriteriaList(
            [StopOnSequence(target_sequences, tokenizer)]
        )
        gemini_client = None
        gemini_types = None
    else:
        torch = None
        transformers = None
        device = None
        tokenizer = None
        model = None
        stopping_criteria = None
        curr_eos = None
        print(
            f"[worker {rank}] Initialising Gemini client for {args.gemini_model} "
            f"({len(shard)} {args.dataset} questions, {len(pending_shard)} pending)."
        )
        gemini_client, gemini_types = make_gemini_client(
            api_key=args.gemini_api_key,
            base_url=args.gemini_base_url,
            api_version=args.gemini_api_version,
        )

    f1_scores = [r["f1"] for r in completed_records]
    em_scores = [r["em"] for r in completed_records]
    start_time = time.time()

    progress = tqdm(
        pending_shard,
        total=len(shard),
        initial=len(completed_records),
        desc=f"worker {rank}",
        position=rank,
        leave=True,
    )

    for item in progress:
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
                    prompt, model, tokenizer, device, stopping_criteria, curr_eos,
                    max_turns=args.max_turns,
                    topk=args.topk,
                    retriever_url=args.retriever_url,
                    max_new_tokens_main=max_main,
                    max_new_tokens_final=max_final,
                )
            else:
                full_output, num_turns = run_inference_gemini(
                    raw_prompt, gemini_client, gemini_types, args.gemini_model,
                    max_turns=args.max_turns,
                    topk=args.topk,
                    retriever_url=args.retriever_url,
                    max_new_tokens_main=max_main,
                    max_new_tokens_final=max_final,
                )

            extracted_answer = extract_answer(full_output)
            if extracted_answer:
                f1 = f1_check(extracted_answer, golden_answers)
                em = em_check(extracted_answer, golden_answers)
            else:
                f1, em = 0.0, 0
                extracted_answer = "[NO_ANSWER]"

        except Exception as exc:
            print(f"\n[worker {rank}] ERROR on question {item['index']}: {exc}")
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
            "worker_rank": rank,
            "backend": args.backend,
            "model": args.model_label,
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if torch is not None and torch.cuda.is_available():
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
        **base_summary_meta,
        "completed_questions": len(f1_scores),
        "avg_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "avg_em": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(rank_summary, f, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# Merge per-rank outputs
# ────────────────────────────────────────────────────────────────────────────

def merge_results(args):
    prefix = DATASETS[args.dataset]["prefix"]
    merged = {}
    for path in sorted(glob(os.path.join(args.output_dir, f"{prefix}_results_rank*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                merged[record["index"]] = record

    merged_records = [merged[idx] for idx in sorted(merged)]
    merged_output_file = os.path.join(args.output_dir, f"{prefix}_results.jsonl")
    with open(merged_output_file, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    f1_scores = [r["f1"] for r in merged_records]
    em_scores = [r["em"] for r in merged_records]
    summary = {
        "dataset": args.dataset,
        "dataset_id": args.dataset_id_resolved,
        "config": args.config,
        "split": args.split,
        "pipeline": args.pipeline,
        "backend": args.backend,
        "model": args.model_label,
        "num_samples": len(merged_records),
        "avg_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "avg_em": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
        "num_workers": args.num_workers,
        "max_turns": args.max_turns,
        "topk": args.topk,
    }
    summary_file = os.path.join(args.output_dir, f"{prefix}_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return merged_output_file, summary_file, summary


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Unified multi-worker EviNote-RAG evaluation")

    p.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()),
                   help="Which dataset to evaluate on")
    p.add_argument("--pipeline", type=str, default="note", choices=list(PIPELINES.keys()),
                   help="Prompt pipeline: note (note-taking) or claim (claim-level)")
    p.add_argument("--backend", type=str, default="local", choices=["local", "gemini"],
                   help="Model backend: local HF causal LM or Gemini-compatible API")

    p.add_argument("--model_id", type=str, default=None,
                   help="HF model ID or local path (required for --backend local)")

    p.add_argument("--gemini_model", type=str, default="Vendor2/Gemini-3-flash",
                   help="Model name to send to the Gemini API")
    p.add_argument("--gemini_api_key", type=str, default=os.environ.get("GEMINI_API_KEY"),
                   help="API key (defaults to $GEMINI_API_KEY)")
    p.add_argument("--gemini_base_url", type=str, default="https://api.gpugeek.com",
                   help="API base URL (e.g. gpugeek gateway)")
    p.add_argument("--gemini_api_version", type=str, default="v1beta",
                   help="HttpOptions api_version")

    p.add_argument("--num_workers", "--num_gpus", dest="num_workers", type=int, default=8,
                   help="For local: # GPUs / one process per GPU. For gemini: # parallel API workers.")

    p.add_argument("--num_samples", type=int, default=None,
                   help="Optional cap on the number of validation questions")
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--dataset_id", type=str, default=None,
                   help="Optional HF dataset ID override")
    p.add_argument("--config", type=str, default=None,
                   help="Dataset config (HotpotQA: distractor/fullwiki; "
                        "TriviaQA: rc.nocontext/...; NQ Open: ignored)")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write per-rank and merged outputs "
                        "(default: ./outputs/eval_{dataset}_{pipeline}_{backend})")

    return p.parse_args()


def main():
    args = parse_args()

    if args.backend == "local" and not args.model_id:
        raise SystemExit("--model_id is required when --backend local")
    if args.backend == "gemini" and not args.gemini_api_key:
        raise SystemExit("--gemini_api_key is required when --backend gemini "
                         "(or set $GEMINI_API_KEY)")

    if args.config is None:
        args.config = DATASETS[args.dataset]["default_config"]

    if args.output_dir is None:
        args.output_dir = f"./outputs/eval_{args.dataset}_{args.pipeline}_{args.backend}"

    args.model_label = args.model_id if args.backend == "local" else args.gemini_model

    if args.backend == "local":
        torch, _ = _local_imports()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --backend local but no GPU was detected.")
        avail = torch.cuda.device_count()
        if args.num_workers > avail:
            raise ValueError(
                f"Requested {args.num_workers} GPUs, but only {avail} are available."
            )

    os.makedirs(args.output_dir, exist_ok=True)

    print("Checking retriever server...")
    search("test query", topk=1, retriever_url=args.retriever_url)
    print("  Retriever is up.")

    label = f"{args.num_samples} samples" if args.num_samples else "full validation set"
    print(f"Loading {args.dataset} ({label}, config={args.config})...")
    loader = DATASETS[args.dataset]["loader"]
    data, dataset_id = loader(
        num_samples=args.num_samples,
        dataset_id=args.dataset_id,
        config=args.config,
        split=args.split,
    )
    args.dataset_id_resolved = dataset_id
    print(f"  Loaded {len(data)} questions from {dataset_id}")

    if args.num_workers <= 1:
        run_worker(0, args, data)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(args.num_workers):
            proc = ctx.Process(target=run_worker, args=(rank, args, data))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            if proc.exitcode and proc.exitcode != 0:
                raise RuntimeError(f"worker exited with code {proc.exitcode}")

    merged_output_file, summary_file, summary = merge_results(args)

    print(f"\n{'=' * 60}")
    print(f"RESULTS  dataset={args.dataset}  pipeline={args.pipeline}  "
          f"backend={args.backend}  ({summary['num_samples']} questions)")
    print(f"  F1:  {summary['avg_f1']:.4f}")
    print(f"  EM:  {summary['avg_em']:.4f}")
    print(f"  Results:  {merged_output_file}")
    print(f"  Summary:  {summary_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
