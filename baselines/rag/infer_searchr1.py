"""
python -m ipdb eval/infer_searchr1.py

Batch inference utilities for SearchR1 / PaperSearchR1.

This module provides a TRANSFORMERS-powered engine for batched inference on
multi-turn, search-augmented prompts (SearchR1-style).

This version uses the standard Hugging Face `transformers` library to ensure
logical consistency with single-instance inference, at the cost of some
performance compared to a vLLM implementation.

The public entry-point is the class `BatchSearchR1` with the method
`run_batch_searchr1(questions)` which returns (full_responses, predictions).

Example
-------
>>> from eval.infer_searchr1 import BatchSearchR1, load_model_and_tokenizer
>>> tokenizer, model = load_model_and_tokenizer("Qwen/Qwen2.5-3B-Instruct")
>>> searchr1 = BatchSearchR1(model=model, tokenizer=tokenizer)
>>> full, preds = searchr1.run_batch_searchr1(["What is the capital of France?", ...])
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple, Optional

import ipdb
import torch
import transformers
import requests
import pandas as pd
from tqdm import tqdm

__all__ = [
    "load_model_and_tokenizer",
    "BatchSearchR1",
]

# -----------------------------------------------------------------------------
# Helper utilities (from infer.py for consistency)
# -----------------------------------------------------------------------------


def get_query(text: str) -> Optional[str]:
    """Extracts the last <search> query from a string."""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def load_model_and_tokenizer(
    model_id: str,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False,
):
    """Load model and tokenizer, optionally from a checkpoint."""
    # ALWAYS load the tokenizer from the original base model ID.
    # The training checkpoint might not contain the complete tokenizer files.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True)

    if checkpoint_path:
        assert os.path.exists(checkpoint_path)
        if verbose:
            print(f"Loading model from checkpoint: {checkpoint_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        if verbose:
            print(f"Loading model from HuggingFace: {model_id}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")

    return tokenizer, model


# -----------------------------------------------------------------------------
# Public convenience wrapper
# -----------------------------------------------------------------------------


class BatchSearchR1:
    """High-level SearchR1 batch inference engine (Transformers-powered)."""

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        search_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = 3,
        max_turns: int = 5,
        max_response_len: int = 1024,
        temperature: float = 0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.search_url = search_url
        self.topk = topk
        self.max_turns = max_turns
        self.max_response_len = max_response_len
        self.temperature = temperature

        # Configure tokenizer for batch generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        print(
            "[BatchSearchR1] Initialised (Transformers-native backend with post-processing)."
        )

    def _batch_search(self, queries: List[str]) -> List[str]:
        """Performs a batched search request to the retrieval server."""
        if not queries:
            return []

        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True
        }
        try:
            response = requests.post(self.search_url, json=payload, timeout=20)
            response.raise_for_status()
            results = response.json().get('result', [])
        except requests.exceptions.RequestException as e:
            print(f"\n❌ ERROR: HTTP request to retrieval server failed: {e}",
                  file=sys.stderr)
            return [""] * len(queries)
        except Exception as e:
            print(f"\n❌ ERROR: Unexpected error during retrieval: {e}",
                  file=sys.stderr)
            return [""] * len(queries)

        def _passages2string(retrieval_result):
            if not retrieval_result:
                return ""
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference

        return [_passages2string(res) for res in results]

    def _postprocess_responses(
            self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(responses,
                                                    skip_special_tokens=True)

        # Truncate at stopping sequences (following training code pattern)
        responses_str = [
            resp.split('</search>')[0] + '</search>'
            if '</search>' in resp else resp.split('</answer>')[0] +
            '</answer>' if '</answer>' in resp else resp
            for resp in responses_str
        ]

        # Re-tokenize the truncated responses
        responses = self.tokenizer(responses_str,
                                   add_special_tokens=False,
                                   return_tensors='pt',
                                   padding="longest")['input_ids']

        return responses, responses_str

    def _build_initial_prompts(self, questions: List[str]) -> List[str]:
        """Constructs the initial system prompts for a batch of questions."""
        prompts = []
        for q in questions:
            q = q.strip()
            if not q.endswith("?"):
                q += "?"
            prompt_str = (
                "Answer the given question. "
                "You must conduct reasoning inside <think> and </think> first every time you get new information. "
                "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
                "and it will return the top searched results between <information> and </information>. "
                "You can search as many times as your want. "
                "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
                "without detailed illustrations. For example, <answer> Beijing </answer>. "
                f"Question: {q}\n")

            if self.tokenizer.chat_template:
                prompt_str = self.tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": prompt_str
                    }],
                    add_generation_prompt=True,
                    tokenize=False)
            prompts.append(prompt_str)
        return prompts

    def run_batch_searchr1(
            self, questions: List[str]) -> Tuple[List[str], List[str]]:
        """
        Run the full, batched SearchR1 loop for a list of questions.
        """
        if not questions:
            return [], []

        # --- Initialization ---
        initial_prompts = self._build_initial_prompts(questions)
        full_traces = {i: p for i, p in enumerate(initial_prompts)}
        active_indices = list(range(len(questions)))

        for turn in range(self.max_turns):
            if not active_indices:
                print(f"[INFO] All traces completed by turn {turn}.")
                break

            # --- 1. Prepare batch for current turn ---
            prompts_for_turn = [full_traces[i] for i in active_indices]

            inputs = self.tokenizer(prompts_for_turn,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=4096).to(self.model.device)

            # --- 2. Batched Generation (simplified - no LogitsProcessor needed) ---
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_response_len,
                temperature=self.temperature,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id)

            # --- 3. Extract only the newly generated tokens ---
            # Remove the input prompt tokens to get only the response
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[:, input_length:]

            # --- 4. Post-process responses to handle stopping ---
            processed_responses, responses_str = self._postprocess_responses(
                response_tokens)

            search_queries_map = {}  # Map from active_idx_in_turn -> query
            finished_in_turn = []

            # ipdb.set_trace()
            # pass

            for i, response_text in enumerate(responses_str):
                original_index = active_indices[i]

                # Add the response to the full trace
                full_traces[original_index] += response_text

                # Check for search queries
                query = get_query(response_text)
                if query:
                    search_queries_map[i] = query

                # Check if this trace is finished (contains </answer>)
                if "</answer>" in response_text:
                    finished_in_turn.append(original_index)

            # --- 5. Batched Retrieval ---
            if search_queries_map:
                indices_that_searched = sorted(search_queries_map.keys())
                queries_to_search = [
                    search_queries_map[i] for i in indices_that_searched
                ]
                search_results = self._batch_search(queries_to_search)

                # --- 6. Append search results to traces ---
                for i, result_text in zip(indices_that_searched,
                                          search_results):
                    original_index = active_indices[i]
                    search_block = f"\n\n<information>{result_text}</information>\n\n"
                    full_traces[original_index] += search_block

            # --- 7. Update active indices ---
            active_indices = [
                i for i in active_indices if i not in finished_in_turn
            ]
            print(
                f"[Turn {turn+1}/{self.max_turns}] Active traces: {len(active_indices)}"
            )

        # --- Finalization ---
        final_traces = [full_traces[i] for i in range(len(questions))]
        predictions = [self._extract_answer(trace) for trace in final_traces]

        return final_traces, predictions

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extracts the last answer from the trace."""
        matches = re.findall(r"<answer>(.*?)</answer>", text,
                             re.DOTALL | re.IGNORECASE)
        return matches[-1].strip() if matches else ""


# -----------------------------------------------------------------------------
# Quick test harness
# -----------------------------------------------------------------------------


def test():
    """Minimal sanity-check when this file is executed directly."""
    import argparse
    from datasets import load_dataset
    from qa_em import compute_score_em
    import time
    import json

    parser = argparse.ArgumentParser(
        description="Quick batch SearchR1 evaluation")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=
        "checkpoints/20250606_papersearchr1v1_qwenit_bm25/global_step_100/",
        help="Path to fine-tuned checkpoint directory")
    parser.add_argument("--model_id",
                        type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model identifier (HF hub or local)")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="jmhb/PaperSearchRL_v4_gv2_n3000_test500",
        help="HF dataset to evaluate on (test split will be used)")
    parser.add_argument(
        "--first_n",
        type=int,
        default=20,
        help="Evaluate only the first N examples for a quick smoke test")

    args = parser.parse_args()

    # Check retrieval server
    try:
        requests.get("http://127.0.0.1:8000/health", timeout=5)
    except Exception as e:
        print(
            "[ERROR] Retrieval server not reachable at http://127.0.0.1:8000 - please start it."
        )
        print(f"Details: {e}")
        sys.exit(1)

    print("[LOAD] Model + tokenizer…")
    tokenizer, model = load_model_and_tokenizer(
        args.model_id, checkpoint_path=args.checkpoint_path, verbose=True)

    print("[LOAD] Dataset…")
    dataset = load_dataset(args.dataset_id, split="test")
    if args.first_n > 0:
        dataset = dataset.select(range(min(args.first_n, len(dataset))))
    questions = dataset["question"]
    gt_answers = dataset[
        "answer"] if "answer" in dataset.column_names else dataset[
            "golden_answers"]

    start_time = time.time()
    print(f"[RUN] Running BatchSearchR1 on {len(questions)} examples…")
    bs_runner = BatchSearchR1(model=model, tokenizer=tokenizer)
    full_traces, predictions = bs_runner.run_batch_searchr1(questions)
    elapsed = time.time() - start_time
    print(
        f"[TIME] Total elapsed: {elapsed:.2f} seconds ({elapsed/len(questions):.2f} s/question)"
    )

    correct = 0
    for pred, gold in zip(predictions, gt_answers):
        gold_list = gold if isinstance(gold, list) else [gold]
        em = compute_score_em(f"<answer></answer><answer>{pred}</answer>",
                              {"target": gold_list})
        if em > 0:
            correct += 1
    acc = correct / len(predictions)
    print(
        f"[RESULT] Exact-match accuracy: {correct}/{len(predictions)} = {acc:.2%}"
    )
    ipdb.set_trace()
    pass

    for i in range(min(3, len(questions))):
        print("\n--- Example", i + 1)
        print("Q:", questions[i])
        print("Pred:", predictions[i])
        print("GT:", gt_answers[i])
        print("Trace snippet:", full_traces[i][:250].replace("\n", " ") + "…")

    with open(f'{args.checkpoint_path}_traces_n_{args.first_n}.json',
              'w') as f:
        json.dump(
            {
                'questions': questions,
                'traces': full_traces,
                'predictions': predictions
            },
            f,
            indent=2)


if __name__ == "__main__":
    test()
