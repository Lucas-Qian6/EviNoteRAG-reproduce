# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
import json

import os
import logging
from verl.utils.entropy.entilement_score import compute_retrival_score_fn
import ray
import numpy as np

logging.basicConfig(level=logging.INFO)

REWARD_MODE_ENV = "EVINOTE_REWARD_MODE"
ROLE_FORMAT_REWARD_WEIGHT = float(os.environ.get("ROLE_FORMAT_REWARD_WEIGHT", "0.05"))
SUMMARY_ENTAILMENT_REWARD_WEIGHT = float(
    os.environ.get(
        "SUMMARY_ENTAILMENT_REWARD_WEIGHT",
        os.environ.get("ANSWER_CLAIM_REWARD_WEIGHT", "0.1"),
    )
)
CLAIM_ENTAILMENT_REWARD_WEIGHT = float(
    os.environ.get(
        "CLAIM_ENTAILMENT_REWARD_WEIGHT",
        os.environ.get("SUMMARY_ENTAILMENT_REWARD_WEIGHT", "0.2"),
    )
)
BRIDGE_REWARD_WEIGHT = float(os.environ.get("BRIDGE_REWARD_WEIGHT", "0.0"))

def get_reward_mode():
    reward_mode = os.environ.get(REWARD_MODE_ENV, "upstream").strip().lower()
    aliases = {
        "original": "upstream",
        "github": "upstream",
        "evinote": "upstream",
        "process": "custom",
        "claim": "custom",
        "claim_level": "custom",
    }
    reward_mode = aliases.get(reward_mode, reward_mode)
    if reward_mode not in {"upstream", "custom"}:
        raise ValueError(
            f"Unsupported {REWARD_MODE_ENV}={reward_mode!r}. "
            "Use 'upstream' or 'custom'."
        )
    return reward_mode


def compute_upstream_retrieval_scores(processed_gts, summaries, retrival_eval_model):
    if retrival_eval_model is None:
        return [0.0] * len(summaries)
    return ray.get(
        retrival_eval_model.batch_retrival_score_fast.remote(
            ground_truths=processed_gts,
            retrival_infos=summaries,
        )
    )

def save_results_to_file(solution_str, answer_content, ground_truths, max_score, data_source=None, f1_score = 0.0, llm_judge_score= 0.0, Search_score= 0.0, retrival_eval_model = None, trajectory_split="train", claim_entailment_score=0.0):
    """Save each training trajectory as one JSONL row."""
    try:
        if trajectory_split == "skip":
            return
        env_key = "EVAL_TRAJECTORY_LOG_FILE" if trajectory_split == "eval" else "TRAIN_TRAJECTORY_LOG_FILE"
        default_file = f"./outputs/eval/{trajectory_split}_trajectories.jsonl"
        json_file_path = os.environ.get(
            env_key,
            default_file,
        )
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        
        # Save data to the new file
        save_json = {
            "solution_str": solution_str,
            "answer_content": answer_content,
            "ground_truths": ground_truths,
            "score": max_score,
            "f1_score": f1_score,
            "llm_judge_score": llm_judge_score,
            "Search_score": Search_score,
            "claim_entailment_score": claim_entailment_score,
            "data_source":data_source,
            "split": trajectory_split,
        }
        json_line = json.dumps(save_json, ensure_ascii=False)
        
        # Write to the JSONL file
        with open(json_file_path, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')
            logging.info(f"File saved: {json_file_path}")
        
            
    except (IOError, OSError) as e:
        logging.error(f"File write failed: {e}")
    except Exception as e:
        logging.error(f"Unknown error: {e}")

  

def check_if_mark_exist(text, keyword_list=["*", "#"]):
    pattern = r'<summary>(.*?)</summary>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return False
    
    stack = []
    is_empty = True # check if notes exists
    for content in matches:
        if content == "":
            continue
        else:
            for char in content:
                if char in keyword_list:
                    is_empty = False
                    if stack and stack[-1] == char:
                        stack.pop()
                    else:
                        stack.append(char)
    
    return len(stack) == 0 and (not is_empty)


def extract_role_claims(summary, role):
    """Extract one-line role-aware claims such as '*Answer* ...' from a summary."""
    if not isinstance(summary, str) or not summary.strip():
        return ""

    role_pattern = re.escape(role)
    matches = re.findall(
        rf'{role_pattern}\s*:?\s*(.*?)(?=\n\s*(?:[-*]\s*)?\*(?:Bridge|Answer)\*|\n\s*-Noise/Uncertain-|\Z)',
        summary,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned_claims = []
    for match in matches:
        claim = re.sub(r'\s+', ' ', match).strip(" -\n\t")
        if claim:
            cleaned_claims.append(claim)

    return " ".join(cleaned_claims)


def has_role_claim(summary, role):
    return bool(extract_role_claims(summary, role))


def has_valid_role_format(summary, answer=None):
    if answer is None or not isinstance(summary, str):
        return False
    has_query_usage_note = bool(
        re.search(r'(?m)^\s*N\d+\s*\([^)]+\)\s*:\s*\S+', summary)
    )
    return has_role_claim(summary, "*Answer*") or has_query_usage_note


def has_valid_claim_workflow_format(solution_str):
    """Check for a relaxed ordered search -> information -> decompose -> relate -> resolve -> summary flow."""
    if not isinstance(solution_str, str) or not solution_str.strip():
        return False

    ordered_patterns = [
        r'<search>.*?</search>',
        r'<information>.*?</information>',
        r'<decompose>.*?</decompose>',
        r'<relate>.*?</relate>',
        r'<resolve>.*?</resolve>',
        r'<summary>.*?</summary>',
    ]
    cursor = 0
    for pattern in ordered_patterns:
        match = re.search(pattern, solution_str[cursor:], re.DOTALL | re.IGNORECASE)
        if not match:
            return False
        cursor += match.end()

    return True


def compute_summary_entailment_scores(summaries, processed_gts, retrival_eval_model):
    """Compute weighted entailment scores for the whole last summary."""
    n = len(summaries)
    summary_entailment_scores = [0.0] * n

    valid_summary_indices = [
        i for i, summary in enumerate(summaries)
        if isinstance(summary, str)
        and summary.strip()
        and str(processed_gts[i]).strip()
    ]
    if retrival_eval_model is not None and SUMMARY_ENTAILMENT_REWARD_WEIGHT > 0 and valid_summary_indices:
        raw_summary_scores = ray.get(
            retrival_eval_model.batch_retrival_score_fast.remote(
                ground_truths=[processed_gts[i] for i in valid_summary_indices],
                retrival_infos=[summaries[i] for i in valid_summary_indices],
            )
        )
        for idx, raw_score in zip(valid_summary_indices, raw_summary_scores):
            summary_entailment_scores[idx] = SUMMARY_ENTAILMENT_REWARD_WEIGHT * raw_score

    return summary_entailment_scores


def extract_information_decompose_pairs(solution_str):
    """Pair each <decompose> block with the most recent preceding <information> block."""
    if not isinstance(solution_str, str) or not solution_str.strip():
        return []

    information_matches = list(
        re.finditer(r'<information>(.*?)</information>', solution_str, re.DOTALL | re.IGNORECASE)
    )
    pairs = []
    for decompose_match in re.finditer(
        r'<decompose>(.*?)</decompose>', solution_str, re.DOTALL | re.IGNORECASE
    ):
        preceding_information = [
            match for match in information_matches if match.end() <= decompose_match.start()
        ]
        if preceding_information:
            pairs.append((preceding_information[-1].group(1).strip(), decompose_match.group(1).strip()))

    return pairs


def parse_information_docs(information):
    """Parse retrieved evidence formatted as Doc 1(Title: ...) text."""
    docs = {}
    if not isinstance(information, str) or not information.strip():
        return docs

    doc_pattern = re.compile(
        r'Doc\s*(\d+)\s*\(Title:\s*(.*?)\)\s*(.*?)(?=\n?Doc\s*\d+\s*\(Title:|$)',
        re.DOTALL | re.IGNORECASE,
    )
    for match in doc_pattern.finditer(information):
        doc_id = match.group(1).strip()
        title = re.sub(r'\s+', ' ', match.group(2)).strip()
        text = re.sub(r'\s+', ' ', match.group(3)).strip()
        docs[doc_id] = f"{title}. {text}" if title else text

    return docs


def parse_decomposed_claims(decompose_block):
    """Extract claims like C1 [Doc1]: claim text, allowing multiline claim text."""
    claims = []
    if not isinstance(decompose_block, str) or not decompose_block.strip():
        return claims

    claim_pattern = re.compile(
        r'^\s*(?:[-*]\s*)?C\d+\s*\[\s*Doc\s*(\d+)\s*\]\s*:\s*(.*?)(?=^\s*(?:[-*]\s*)?C\d+\s*\[\s*Doc\s*\d+\s*\]\s*:|\Z)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    for match in claim_pattern.finditer(decompose_block):
        doc_id = match.group(1).strip()
        claim = re.sub(r'\s+', ' ', match.group(2)).strip()
        if claim:
            claims.append((doc_id, claim))

    return claims


def build_claim_entailment_pairs(solution_str):
    """Build (claim, source evidence) pairs from all information/decompose turns."""
    claim_evidence_pairs = []
    for information, decompose_block in extract_information_decompose_pairs(solution_str):
        docs = parse_information_docs(information)
        fallback_evidence = re.sub(r'\s+', ' ', information).strip()
        for doc_id, claim in parse_decomposed_claims(decompose_block):
            evidence = docs.get(doc_id, fallback_evidence)
            if claim and evidence:
                claim_evidence_pairs.append((claim, evidence))

    return claim_evidence_pairs


def compute_claim_decompose_entailment_scores(solution_strs, retrival_eval_model):
    """Reward decomposed claims that are entailed by their source evidence."""
    n = len(solution_strs)
    claim_entailment_scores = [0.0] * n
    if retrival_eval_model is None or CLAIM_ENTAILMENT_REWARD_WEIGHT <= 0:
        return claim_entailment_scores

    flat_claims = []
    flat_evidences = []
    index_mapping = []
    for idx, solution_str in enumerate(solution_strs):
        for claim, evidence in build_claim_entailment_pairs(solution_str):
            index_mapping.append(idx)
            flat_claims.append(claim)
            flat_evidences.append(evidence)

    if not flat_claims:
        return claim_entailment_scores

    raw_scores = ray.get(
        retrival_eval_model.batch_retrival_score_fast.remote(
            ground_truths=flat_claims,
            retrival_infos=flat_evidences,
        )
    )

    grouped_scores = [[] for _ in range(n)]
    for idx, raw_score in zip(index_mapping, raw_scores):
        grouped_scores[idx].append(raw_score)

    for idx, scores in enumerate(grouped_scores):
        if scores:
            claim_entailment_scores[idx] = CLAIM_ENTAILMENT_REWARD_WEIGHT * (
                sum(scores) / len(scores)
            )

    return claim_entailment_scores




def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def f1_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    def compute_f1_score(prediction_tokens, golden_tokens):
        common = set(prediction_tokens) & set(golden_tokens)
        num_same = len(common)
        if num_same == 0:
            return 0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(golden_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    normalized_prediction = normalize_answer(prediction).split()
    max_f1 = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer).split()
        f1 = compute_f1_score(normalized_prediction, golden_answer)
        # print(f1)
        if f1 > max_f1:
            max_f1 = f1
    
    return max_f1
    

def compute_score_f1(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    The scoring function for F1 score.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        f1_value = f1_check(answer, ground_truth['target'])
        if f1_value > 0:
            return score * f1_value
        else:
            return format_score



@ray.remote
def batched_compute_score_f1_ver(
    solution_strs, 
    ground_truths, 
    summaries,  
    processed_gts,
    data_sources,
    format_score=0., 
    score=1., 
    retrival_eval_model=None,
    trajectory_split="train",
):
    """
    Batch process multiple samples to avoid single-instance model inference.
    
    Args:
        solution_strs: List[str], multiple decoded texts
        ground_truths: List[str], list of preprocessed ground truth strings
        summaries: List[str], summaries extracted from solution_str
        processed_gts: List[str], preprocessed answers (e.g., normalized answers)
        format_score: float, penalty score for formatting errors
        score: float, base score for correct answers
        retrival_eval_model: Ray remote actor, supports batch_compute_score method
    
    Returns:
        scores: List[float], the final score for each sample
    """
    # Data length check
    assert len(solution_strs) == len(ground_truths) == len(summaries), "All lists must be the same length"

    reward_mode = get_reward_mode()
    if reward_mode == "custom":
        claim_entailment_scores = compute_claim_decompose_entailment_scores(
            solution_strs=solution_strs,
            retrival_eval_model=retrival_eval_model,
        )
    else:
        batch_retrival_scores = compute_upstream_retrieval_scores(
            processed_gts=processed_gts,
            summaries=summaries,
            retrival_eval_model=retrival_eval_model,
        )

    scores = []
    for i, (solution_str, ground_truth, summary, data_source) in enumerate(zip(solution_strs, ground_truths, summaries, data_sources)):
        answer = extract_solution(solution_str)

        rd_score = 0.
        f1_value = 0.
        process_score = 0.
        search_score = 0.
        format_score_value = 0.0
        claim_entailment_score = 0.0

        if reward_mode == "upstream":
            if answer is None or summary is None:
                rd_score = 0
            else:
                f1_value = f1_check(answer, ground_truth)
                if f1_value > 0:
                    rd_score = score * f1_value
                elif check_if_mark_exist(solution_str):
                    rd_score = 0.1
                else:
                    rd_score = format_score
            search_score = batch_retrival_scores[i]
            rd_score += search_score
        else:
            claim_entailment_score = claim_entailment_scores[i]
            if answer is not None:
                if has_valid_claim_workflow_format(solution_str):
                    format_score_value = ROLE_FORMAT_REWARD_WEIGHT
                f1_value = f1_check(answer, ground_truth)
                if f1_value > 0:
                    rd_score = score * f1_value
                else:
                    rd_score = format_score
            else:
                rd_score = 0
            process_score = format_score_value + claim_entailment_score
            search_score = claim_entailment_score
            rd_score += process_score

        save_results_to_file(
            solution_str,
            answer,
            ground_truth,
            rd_score,
            data_source=data_source,
            f1_score=f1_value,
            llm_judge_score=process_score,
            Search_score=search_score,
            trajectory_split=trajectory_split,
            claim_entailment_score=claim_entailment_score,
        )
        if random.randint(1, 64) == 1:
            print(f"-------------- [train] ----------------")
            print(f"Solution string: {solution_str}")
            print(f"Extracted summary: {summary}")
            print(f"Extracted answer: {answer}")
            print(f"Golden answers: {ground_truth}")
            print(f"Score: {rd_score}")
            print(f"F1 value: {f1_value}")
            print(f"Reward_mode: {reward_mode}")
            if reward_mode == "custom":
                print(f"Format_score: {format_score_value}")
                print(f"Claim_entailment_score: {claim_entailment_score}")
            else:
                print(f"Search_score: {search_score}")
            print(f"data_source: {data_source}")
            

        scores.append(rd_score)

    return scores


@ray.remote
def batched_compute_score_em_ver(
    solution_strs, 
    ground_truths, 
    summaries,  
    processed_gts,
    data_sources,
    format_score=0., 
    score=1., 
    retrival_eval_model=None,
    trajectory_split="train",
):
    """
    Batch process multiple samples to avoid single-instance model inference.
    
    Args:
        solution_strs: List[str], multiple decoded texts
        ground_truths: List[str], list of preprocessed ground truth strings
        summaries: List[str], summaries extracted from solution_str
        processed_gts: List[str], preprocessed answers (e.g., normalized answers)
        format_score: float, penalty score for formatting errors
        score: float, base score for correct answers
        retrival_eval_model: Ray remote actor, supports batch_compute_score method
    
    Returns:
        scores: List[float], the final score for each sample
    """
    # Data length check
    assert len(solution_strs) == len(ground_truths) == len(summaries), "All lists must be the same length"

    reward_mode = get_reward_mode()
    if reward_mode == "custom":
        claim_entailment_scores = compute_claim_decompose_entailment_scores(
            solution_strs=solution_strs,
            retrival_eval_model=retrival_eval_model,
        )
    else:
        batch_retrival_scores = compute_upstream_retrieval_scores(
            processed_gts=processed_gts,
            summaries=summaries,
            retrival_eval_model=retrival_eval_model,
        )

    scores = []
    for i, (solution_str, ground_truth, summary, data_source) in enumerate(zip(solution_strs, ground_truths, summaries, data_sources)):
        answer = extract_solution(solution_str)

        rd_score = 0.
        process_score = 0.
        search_score = 0.
        format_score_value = 0.0
        claim_entailment_score = 0.0

        if reward_mode == "upstream":
            if answer is None or summary is None:
                rd_score = 0
            else:
                if em_check(answer, ground_truth):
                    rd_score = score # 1.0
                elif check_if_mark_exist(solution_str):
                    rd_score = 0.1
                else:
                    rd_score = format_score # 0.0
            search_score = batch_retrival_scores[i]
            rd_score += search_score
        else:
            claim_entailment_score = claim_entailment_scores[i]
            if answer is not None:
                if has_valid_claim_workflow_format(solution_str):
                    format_score_value = ROLE_FORMAT_REWARD_WEIGHT
                if em_check(answer, ground_truth):
                    rd_score = score # 1.0
                else:
                    rd_score = format_score # 0.0
            else:
                rd_score = 0
            process_score = format_score_value + claim_entailment_score
            search_score = claim_entailment_score
            rd_score += process_score

        save_results_to_file(
            solution_str,
            answer,
            ground_truth,
            rd_score,
            data_source=data_source,
            llm_judge_score=process_score,
            Search_score=search_score,
            trajectory_split=trajectory_split,
            claim_entailment_score=claim_entailment_score,
        )
        if random.randint(1, 64) == 1:
            print(f"-------------- [train] ----------------")
            print(f"Solution string: {solution_str}")
            print(f"Extracted summary: {summary}")
            print(f"Extracted answer: {answer}")
            print(f"Golden answers: {ground_truth}")
            print(f"Score: {rd_score}")
            print(f"Reward_mode: {reward_mode}")
            if reward_mode == "custom":
                print(f"Format_score: {format_score_value}")
                print(f"Claim_entailment_score: {claim_entailment_score}")
            else:
                print(f"Search_score: {search_score}")
            print(f"data_source: {data_source}")
            

        scores.append(rd_score)

    return scores
