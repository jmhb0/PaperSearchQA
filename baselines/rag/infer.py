"""
python -m ipdb eval/infer.py

SearchR1 inference script 

Standard checkpoints from OG repo 
    PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo
    PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo

    
python -m ipdb eval/infer.py --model_id PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo --first_n 300 --dataset_name jmhb/PaperSearchRL_v4_gv2_n3000_test500

SearchR1 pap
"""
import transformers
import torch
import time
import random
from datasets import load_dataset
import requests
import os
from pathlib import Path
import ipdb
import pandas as pd
from tqdm import tqdm
from qa_em import compute_score_em
import re
import sys
import argparse
import hashlib
import json
import lmdb
import atexit
from filelock import FileLock
from typing import Optional
from dataclasses import dataclass, asdict

results_dir = "results/eval_inference_with_search"
Path(results_dir).mkdir(parents=True, exist_ok=True)

curr_eos = [151645, 151643]  # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

PORT = 8001

@dataclass
class SearchR1Config:
    """Configuration for SearchR1 caching."""
    model_id: str
    checkpoint_path: Optional[str] = None
    retriever_type: str = "bm25"
    corpus_filename: str = "pubmed.jsonl"
    temperature: float = 0.7


class SearchR1Cache:
    """LMDB-based cache for SearchR1 results."""

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize the cache with LMDB."""
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, "searchr1_cache.lmdb")
        self.lock_path = self.cache_path + ".lock"
        self.lock = FileLock(self.lock_path)
        self.env = lmdb.open(self.cache_path,
                             map_size=1024 * 1024 * 1024)  # 1GB max

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def _generate_model_signature(self, config: SearchR1Config) -> str:
        """Generate model signature from model_id and checkpoint_path."""
        if config.checkpoint_path:
            return f"{config.model_id}|{config.checkpoint_path}"
        else:
            return config.model_id

    def _generate_key(self, config: SearchR1Config, input_prompt: str) -> str:
        """Generate a unique cache key based on config and input prompt."""
        # Create model signature
        model_signature = self._generate_model_signature(config)

        # Create cache key components
        key_components = {
            "model_signature": model_signature,
            "input_prompt": input_prompt,
            "retriever_type": config.retriever_type,
            "corpus_filename": config.corpus_filename,
            "temperature": config.temperature
        }

        # Create a hash of the key components
        content = json.dumps(key_components, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, config: SearchR1Config, input_prompt: str) -> Optional[str]:
        """Retrieve cached result if it exists."""
        key = self._generate_key(config, input_prompt)

        with self.env.begin() as txn:
            cached_value = txn.get(key.encode())
            if cached_value:
                print("✓", end="", flush=True)  # Cache hit indicator
                return cached_value.decode()
        return None

    def set(self, config: SearchR1Config, input_prompt: str, result: str):
        """Cache the result."""
        key = self._generate_key(config, input_prompt)

        with self.lock:
            with self.env.begin(write=True) as txn:
                txn.put(key.encode(), result.encode())

    def close(self):
        """Close the LMDB environment."""
        self.env.close()


# Global cache instance
_searchr1_cache = SearchR1Cache()
atexit.register(_searchr1_cache.close)


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):

    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [
            torch.as_tensor(target_id, device=input_ids.device)
            for target_id in self.target_ids
        ]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False


def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def search(query: str,
           topk: int = 3,
           retriever_type: str = "bm25",
           corpus_filename: str = "pubmed.jsonl"):
    payload = {"queries": [query], "topk": topk, "return_scores": True}

    try:
        response = requests.post(f"http://127.0.0.1:{PORT}/retrieve",
                                 json=payload,
                                 timeout=10)
        response.raise_for_status()
        response_data = response.json()
        results = response_data['result']

        # Validate retriever type matches
        server_retriever_type = response_data['retriver_type']
        if server_retriever_type != retriever_type:
            raise ValueError(
                f"Retriever type mismatch. Expected: {retriever_type}, Server: {server_retriever_type}"
            )

        # Validate corpus filename matches
        server_corpus_filename = response_data['corpus_filename']
        if server_corpus_filename != corpus_filename:
            raise ValueError(
                f"Corpus filename mismatch. Expected: {corpus_filename}, Server: {server_corpus_filename}"
            )

    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Failed to connect to retrieval server at localhost:8000: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during retrieval: {e}")

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):

            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


def load_model_and_tokenizer(model_id, checkpoint_path=None, verbose=0):
    """Load model and tokenizer, optionally from a checkpoint."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

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


def inference_with_search(question,
                          tokenizer,
                          model,
                          retriever_type="bm25",
                          corpus_filename="pubmed.jsonl",
                          temperature=0.7,
                          verbose=0,
                          model_id=None,
                          checkpoint_path=None,
                          use_cache=True,
                          overwrite_cache=False):
    """Run inference with search for a single question and return the full trace."""
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    # Prepare the initial message
    initial_prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    # Apply chat template if available
    if tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": initial_prompt
            }],
            add_generation_prompt=True,
            tokenize=False)
    else:
        formatted_prompt = initial_prompt

    # Create config for caching (if model_id is provided)
    if model_id is not None and use_cache:
        cache_config = SearchR1Config(model_id=model_id,
                                      checkpoint_path=checkpoint_path,
                                      retriever_type=retriever_type,
                                      corpus_filename=corpus_filename,
                                      temperature=temperature)

        # Check cache first (unless overwriting)
        if not overwrite_cache:
            cached_result = _searchr1_cache.get(cache_config, formatted_prompt)
            if cached_result:
                if verbose:
                    print("Using cached result")
                return cached_result

    # Initialize the stopping criteria
    target_sequences = [
        "</search>", " </search>", "</search>\n", " </search>\n",
        "</search>\n\n", " </search>\n\n", "?</search>", "?</search>\n",
        "?</search>\n\n"
    ]
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)])

    cnt = 0
    full_trace = ""
    prompt = formatted_prompt

    if verbose:
        print(
            '\n\n################# [Start Reasoning + Searching] ##################\n\n'
        )
        print(prompt)
    full_trace += prompt

    # Encode the chat-formatted prompt
    # Let device_map="auto" handle device placement automatically
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        # Move to the same device as the model's first parameter
        if hasattr(model, 'device'):
            input_ids = input_ids.to(model.device)
        else:
            # For models with device_map="auto", use the device of the first parameter
            input_ids = input_ids.to(next(model.parameters()).device)

        attention_mask = torch.ones_like(input_ids)

        # Generate text with the stopping criteria
        if temperature > 0:
            # Sampling-based generation
            outputs = model.generate(input_ids,
                                     attention_mask=attention_mask,
                                     max_new_tokens=1024,
                                     stopping_criteria=stopping_criteria,
                                     pad_token_id=tokenizer.eos_token_id,
                                     do_sample=True,
                                     temperature=temperature)
        else:
            # Deterministic generation
            outputs = model.generate(input_ids,
                                     attention_mask=attention_mask,
                                     max_new_tokens=1024,
                                     stopping_criteria=stopping_criteria,
                                     pad_token_id=tokenizer.eos_token_id,
                                     do_sample=False,
                                     top_p=None,
                                     top_k=None,
                                     temperature=None)

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens,
                                           skip_special_tokens=True)
            if verbose:
                print(output_text)
            full_trace += output_text
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens,
                                       skip_special_tokens=True)

        tmp_query = get_query(
            tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print(f'searching "{tmp_query}"...')
            search_results = search(tmp_query,
                                    retriever_type=retriever_type,
                                    corpus_filename=corpus_filename)
        else:
            search_results = ''

        search_text = curr_search_template.format(
            output_text=output_text, search_results=search_results)
        prompt += search_text
        full_trace += search_text
        cnt += 1
        if verbose:
            print(search_text)

    # Cache the result if caching is enabled and model_id is provided
    if model_id is not None and (use_cache or overwrite_cache):
        _searchr1_cache.set(cache_config, formatted_prompt, full_trace)

    return full_trace


def extract_answer(text):
    """Extract answer from <answer>...</answer> tags."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return ""


def eval_inference_with_search(model_id,
                               dataset_name,
                               first_n,
                               retriever_type="bm25",
                               corpus_filename="pubmed.jsonl",
                               temperature=0.0,
                               checkpoint_path=None,
                               verbose=0,
                               use_cache=True,
                               overwrite_cache=False):
    """Evaluate inference with search following the structure of eval_direct_inference.py"""
    if verbose:
        print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(model_id,
                                                checkpoint_path,
                                                verbose=verbose)

    if verbose:
        print("Loading dataset...")
    # Load the test dataset
    dataset = load_dataset(dataset_name, split="test")

    # Apply first_n limit if specified
    if first_n > 0 and len(dataset) > first_n:
        dataset = dataset.select(range(first_n))

    # Convert to DataFrame
    df = dataset.to_pandas()
    if verbose:
        print(f"Loaded {len(df)} test examples")

    # Initialize results
    results = []

    if verbose:
        print("Running inference with search...")
    for idx, row in tqdm(df.iterrows(),
                         total=len(df),
                         desc="Processing questions"):
        question = row['question']
        golden_answers = row['golden_answers']

        # Run inference with search (now with caching support)
        output_text = inference_with_search(question,
                                            tokenizer,
                                            model,
                                            retriever_type=retriever_type,
                                            corpus_filename=corpus_filename,
                                            temperature=temperature,
                                            verbose=verbose,
                                            model_id=model_id,
                                            checkpoint_path=checkpoint_path,
                                            use_cache=use_cache,
                                            overwrite_cache=overwrite_cache)

        # Extract answer
        extracted_answer = extract_answer(output_text)

        # Format ground truth for compute_score_em function
        # The function expects a dict with 'target' key containing the list of golden answers
        ground_truth = {'target': golden_answers}

        # Compute correctness score
        score = compute_score_em(
            solution_str=output_text,  # Pass the full output text for extraction
            ground_truth=ground_truth,
            method='strict',
            format_score=0.0,
            score=1.0)

        # Store results
        result = {
            'question': question,
            'golden_answers': golden_answers,
            'generated_text': output_text,
            'extracted_answer': extracted_answer,
            'correctness_score': score
        }
        results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def save_results(results_df,
                 model_id,
                 dataset_name,
                 first_n=0,
                 checkpoint_path=None,
                 verbose=0):
    """Save results with descriptive filenames including model, dataset, and first_n info."""
    # Create safe dataset name for folder
    safe_dataset_name = dataset_name.replace("/", "_")

    # Create directory name based on checkpoint path if available, otherwise use model_id
    if checkpoint_path:
        # Remove "checkpoints", "actor", "global_" and replace "/" with "-"
        dir_name = checkpoint_path.replace("checkpoints/", "").replace(
            "actor/", "").replace("global_", "").replace("/", "-")
        # Remove trailing slash if present
        dir_name = dir_name.rstrip("/")
    else:
        dir_name = model_id.replace("/", "_")

    # Create dataset-specific directory
    results_dir = Path("results/infer") / safe_dataset_name / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Calculate and print mean correctness score
    mean_score = results_df['correctness_score'].mean()
    print(f"\nMean correctness score: {mean_score:.4f}")
    print(
        f"Total correct answers: {(results_df['correctness_score'] > 0).sum()}/{len(results_df)}"
    )

    # Create safe filename components by replacing "/" with "_"
    safe_model_id = model_id.replace("/", "_")

    # Build filename components (model_id is primary)
    filename_parts = [safe_model_id]
    if first_n > 0:
        filename_parts.append(f"first_{first_n}")

    base_filename = "_".join(filename_parts)

    # Save results to CSV
    output_path = results_dir / f"{base_filename}_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Save summary statistics as readable text file
    summary_stats = {
        'Dataset':
        dataset_name,
        'Model':
        model_id,
        'Checkpoint Path':
        checkpoint_path if checkpoint_path else "None",
        'First N Examples':
        first_n if first_n > 0 else "All",
        'Total Questions':
        len(results_df),
        'Correct Answers': (results_df['correctness_score'] > 0).sum(),
        'Mean Correctness Score':
        f"{mean_score:.4f}",
        'Accuracy Percentage':
        f"{(results_df['correctness_score'] > 0).mean() * 100:.2f}%"
    }

    summary_path = results_dir / f"{base_filename}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

    print(f"Summary statistics saved to: {summary_path}")

    return results_df


def eval_dataset():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate inference with search")

    parser.add_argument("--model_id",
                        type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model ID or path")
    parser.add_argument("--dataset_name",
                        type=str,
                        default="jmhb/PaperSearchRL_v4_gv2_n3000_test500",
                        help="Dataset name")
    parser.add_argument("--first_n",
                        type=int,
                        default=0,
                        help="Number of examples to process (0 for all)")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.0,
                        help="Temperature for generation (default: 0.0)")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        # default="checkpoints/20250606_papersearchr1v1_qwenit_bm25/global_step_100/",
        help=
        "Path to model checkpoint (default: checkpoints/20250606_papersearchr1v1_qwenit_bm25/global_step_100/)"
    )
    parser.add_argument("--retriever_type",
                        type=str,
                        default="bm25",
                        choices=["bm25", "e5"],
                        help="Type of retriever to use (default: bm25)")
    parser.add_argument(
        "--corpus_filename",
        type=str,
        default="pubmed.jsonl",
        help="Expected corpus filename (default: pubmed.jsonl)")
    parser.add_argument("--verbose",
                        type=int,
                        default=0,
                        help="Verbosity level")

    # Cache-related arguments
    parser.add_argument("--no_cache",
                        action="store_true",
                        help="Disable caching")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help=
        "Skip reading from cache but write new results to cache (overwrite existing)"
    )

    args = parser.parse_args()

    # Run evaluation
    df = eval_inference_with_search(model_id=args.model_id,
                                    dataset_name=args.dataset_name,
                                    first_n=args.first_n,
                                    retriever_type=args.retriever_type,
                                    corpus_filename=args.corpus_filename,
                                    temperature=args.temperature,
                                    checkpoint_path=args.checkpoint_path,
                                    verbose=args.verbose,
                                    use_cache=not args.no_cache,
                                    overwrite_cache=args.overwrite_cache)

    save_results(df,
                 model_id=args.model_id,
                 dataset_name=args.dataset_name,
                 first_n=args.first_n,
                 checkpoint_path=args.checkpoint_path,
                 verbose=args.verbose)


def eval_one_question(
        question,
        model_id="Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path="checkpoints/20250606_papersearchr1v1_qwenit_bm25/global_step_100/",
        temperature=0.0,
        retriever_type="bm25",
        corpus_filename="pubmed.jsonl",
        verbose=1,
        use_cache=True,
        overwrite_cache=False):
    """
    Evaluate a single hardcoded question using inference with search.
    Uses default values from the argparse configuration.
    """

    if verbose:
        print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(model_id,
                                                checkpoint_path,
                                                verbose=verbose)

    if verbose:
        print(f"Question: {question}")
        print("Running inference with search...")

    # Run inference with search
    output_text = inference_with_search(question,
                                        tokenizer,
                                        model,
                                        retriever_type=retriever_type,
                                        corpus_filename=corpus_filename,
                                        temperature=temperature,
                                        verbose=verbose,
                                        model_id=model_id,
                                        checkpoint_path=checkpoint_path,
                                        use_cache=use_cache,
                                        overwrite_cache=overwrite_cache)

    # Extract answer
    extracted_answer = extract_answer(output_text)

    if verbose:
        print(f"\nExtracted Answer: {extracted_answer}")

    return {
        'question': question,
        'generated_text': output_text,
        'extracted_answer': extracted_answer
    }


if __name__ == "__main__":
    RUN_ONE_QUESTION = False

    if not RUN_ONE_QUESTION:
        df = eval_dataset()
    else:
        question = "What congenital brain abnormality is characterized by obstruction of cerebrospinal fluid flow due to atresia of one foramen of Monro?"
        question = "Obstruction at which anatomical structure is commonly implicated in the development of unilateral hydrocephalus?"
        result = eval_one_question(question)
        print(result['generated_text'])
