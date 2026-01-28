"""
python -m ipdb eval/run_inference.py --method rag --first_n 300 --model_id qwen/Qwen2.5-3B-Instruct --dataset_id jmhb/PaperSearchRL_v4_gv2_n3000_test500
python -m ipdb eval/run_inference.py --method rag --first_n 300 --model_id qwen/Qwen2.5-7B-Instruct --dataset_id jmhb/PaperSearchRL_v4_gv2_n3000_test500

Inference script for PaperSearchRL evaluation with multiple methods.
Supports: Direct Inference, CoT, RAG, SearchR1, and PaperSearchR1.
The RAG, SearchR1, and PaperSearchR1 methods assume a running server on 127.0.0.1:8001.
"""

import os
import sys
import hashlib
import json
import argparse
import lmdb
import atexit
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import transformers
import torch
import numpy as np
import requests
from qa_em import compute_score_em
import re
import ipdb
from filelock import FileLock

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_generation.core_pipeline.api import call_llm, call_llm_batch

# Local imports
from batch_rag import BatchRAG
from infer_searchr1 import BatchSearchR1
from infer import load_model_and_tokenizer, inference_with_search

# VLLM imports for fast inference
from vllm import LLM, SamplingParams


@dataclass
class InferenceConfig:
    """Configuration for inference methods."""
    method: str
    model_id: str
    dataset_id: str
    max_tokens: int = 1024
    temperature: float = 0.7
    batch_size: int = 1
    first_n: int = 10000
    # Method-specific configs
    rag_top_k: int = 3
    rag_retriever_type: Optional[str] = "bm25"
    rag_corpus_filename: str = "pubmed.jsonl"
    use_cache: bool = True
    overwrite_cache: bool = False
    # VLLM specific configs
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 4096


class InferenceCache:
    """LMDB-based cache for inference results."""

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize the cache with LMDB."""
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, "infer_cache.lmdb")
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

    def _generate_key(self, config: InferenceConfig, input_data: str) -> str:
        """Generate a unique cache key based on config and input."""
        # Create a hash of the config and input
        config_str = json.dumps(asdict(config), sort_keys=True)
        content = f"{config_str}|{input_data}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, config: InferenceConfig, input_data: str) -> Optional[str]:
        """Retrieve cached result if it exists."""
        key = self._generate_key(config, input_data)

        with self.env.begin() as txn:
            cached_value = txn.get(key.encode())
            if cached_value:
                print("✓", end="", flush=True)  # Cache hit indicator
                return cached_value.decode()
        return None

    def set(self, config: InferenceConfig, input_data: str, result: str):
        """Cache the result."""
        key = self._generate_key(config, input_data)

        with self.lock:
            with self.env.begin(write=True) as txn:
                txn.put(key.encode(), result.encode())

    def close(self):
        """Close the LMDB environment."""
        self.env.close()


# Global cache instance
_inference_cache = InferenceCache()
atexit.register(_inference_cache.close)


class InferenceEngine:
    """Main inference engine supporting multiple methods."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.vllm_model = None
        self.tokenizer = None
        self.searchr1_model = None
        self.searchr1_tokenizer = None
        self.batch_rag = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""

        # For other methods, use VLLM
        self.vllm_model = LLM(
            model=self.config.model_id,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            max_model_len=self.config.vllm_max_model_len,
            trust_remote_code=True)
        print(f"Loaded VLLM model: {self.config.model_id}")

        # Load tokenizer for all VLLM-based methods
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_id)
        print(f"Loaded tokenizer: {self.config.model_id}")

        # Initialize BatchRAG for RAG method
        if self.config.method == "rag":
            self.batch_rag = BatchRAG(
                vllm_model=self.vllm_model,
                tokenizer=self.tokenizer,
                retrieval_topk=self.config.rag_top_k,
                retriever_type=self.config.rag_retriever_type,
                corpus_filename=self.config.rag_corpus_filename)
            print(
                f"Loaded BatchRAG with top-k={self.config.rag_top_k}, retriever={self.config.rag_retriever_type}, corpus={self.config.rag_corpus_filename}"
            )

    def _generate_response(self, prompt: str) -> str:
        """Generate response for a single prompt using VLLM."""
        return self._generate_responses_batch([prompt])[0]

    def _generate_responses_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts at once using VLLM's batching."""
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stop=["<|im_end|>", "</s>", "<|endoftext|>"])

        outputs = self.vllm_model.generate(prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def _construct_prompt(self,
                          question: str,
                          context: str = "",
                          method: str = "direct") -> str:
        """Construct prompt based on method."""
        prompt_parts = []

        if context:
            prompt_parts.append(f"Context: {context}")

        if method == "direct":
            prompt_parts.extend([
                f"Question: {question}", "",
                "Instructions: Provide a direct answer to the question between the tags <answer> and </answer>. Your answer should be a single entity, fact, or short phrase. Do not provide explanations or reasoning.",
                "", "Answer:"
            ])
        elif method == "cot":
            prompt_parts.extend([
                f"Question: {question}", "",
                "Instructions: Think through this question step by step. After your reasoning, provide your final answer as a single entity, fact, or short phrase enclosed in <answer> and </answer> tags.",
                "", "Let's think step by step:", ""
            ])
        else:
            raise ValueError(
                f"Unknown method for prompt construction: {method}")

        return "\n".join(prompt_parts)

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template to prompt if available."""
        if self.tokenizer and self.tokenizer.chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": prompt
                }],
                add_generation_prompt=True,
                tokenize=False)
            return formatted_prompt
        else:
            return prompt

    def direct_inference_batch(
            self,
            questions: List[str],
            contexts: List[str] = None) -> List[Tuple[str, str]]:
        """Batch direct inference for multiple questions."""
        if contexts is None:
            contexts = [""] * len(questions)

        # Construct all prompts first
        raw_prompts = [
            self._construct_prompt(q, c, "direct")
            for q, c in zip(questions, contexts)
        ]

        # Apply chat template to all prompts
        prompts = [self._apply_chat_template(prompt) for prompt in raw_prompts]

        # Check cache for all items first (use formatted prompts for cache keys)
        results = []
        uncached_indices = []
        uncached_prompts = []

        for i, prompt in enumerate(prompts):
            if self.config.use_cache and not self.config.overwrite_cache:
                cached_result = _inference_cache.get(self.config, prompt)
                if cached_result:
                    prediction = self._extract_prediction(cached_result)
                    results.append((cached_result, prediction))
                    continue

            # Not cached, need to generate
            uncached_indices.append(i)
            uncached_prompts.append(prompt)
            results.append(None)  # Placeholder

        # Generate responses for uncached items in batch
        if uncached_prompts:
            print(f"Generating {len(uncached_prompts)} uncached responses...",
                  end="",
                  flush=True)
            responses = self._generate_responses_batch(uncached_prompts)

            # Fill in results and cache
            for idx, response in zip(uncached_indices, responses):
                prediction = self._extract_prediction(response)
                results[idx] = (response, prediction)

                # Cache the result using formatted prompt
                if self.config.use_cache or self.config.overwrite_cache:
                    prompt = prompts[idx]
                    _inference_cache.set(self.config, prompt, response)

        return results

    def cot_inference_batch(
            self,
            questions: List[str],
            contexts: List[str] = None) -> List[Tuple[str, str]]:
        """Batch CoT inference for multiple questions."""
        if contexts is None:
            contexts = [""] * len(questions)

        # Construct all prompts first
        raw_prompts = [
            self._construct_prompt(q, c, "cot")
            for q, c in zip(questions, contexts)
        ]

        # Apply chat template to all prompts
        prompts = [self._apply_chat_template(prompt) for prompt in raw_prompts]

        # Check cache for all items first (use formatted prompts for cache keys)
        results = []
        uncached_indices = []
        uncached_prompts = []

        for i, prompt in enumerate(prompts):
            if self.config.use_cache and not self.config.overwrite_cache:
                cached_result = _inference_cache.get(self.config, prompt)
                if cached_result:
                    prediction = self._extract_prediction(cached_result)
                    results.append((cached_result, prediction))
                    continue

            # Not cached, need to generate
            uncached_indices.append(i)
            uncached_prompts.append(prompt)
            results.append(None)  # Placeholder

        # Generate responses for uncached items in batch
        if uncached_prompts:
            print(f"Generating {len(uncached_prompts)} uncached responses...",
                  end="",
                  flush=True)
            responses = self._generate_responses_batch(uncached_prompts)

            # Fill in results and cache
            for idx, response in zip(uncached_indices, responses):
                prediction = self._extract_prediction(response)
                results[idx] = (response, prediction)

                # Cache the result using formatted prompt
                if self.config.use_cache or self.config.overwrite_cache:
                    prompt = prompts[idx]
                    _inference_cache.set(self.config, prompt, response)

        return results

    def rag_inference_batch(
            self,
            questions: List[str],
            contexts: List[str] = None) -> List[Tuple[str, str]]:
        """Batch RAG inference using BatchRAG."""
        if self.batch_rag is None:
            raise RuntimeError("BatchRAG not initialized properly")

        # Check cache for all items first
        results = []
        uncached_indices = []
        uncached_questions = []

        for i, question in enumerate(questions):
            cache_key = f"rag:{question}|"
            if self.config.use_cache and not self.config.overwrite_cache:
                cached_result = _inference_cache.get(self.config, cache_key)
                if cached_result:
                    prediction = self._extract_prediction(cached_result)
                    results.append((cached_result, prediction))
                    continue

            # Not cached, need to generate
            uncached_indices.append(i)
            uncached_questions.append(question)
            results.append(None)  # Placeholder

        # Generate responses for uncached items using batch RAG
        if uncached_questions:
            print(
                f"Running batch RAG for {len(uncached_questions)} uncached questions...",
                end="",
                flush=True)
            try:
                contexts, generated_texts, prompts = self.batch_rag.run_batch_rag(
                    uncached_questions,
                    retriever_type=self.config.rag_retriever_type,
                    corpus_filename=self.config.rag_corpus_filename)

                # Fill in results and cache
                for idx, response in zip(uncached_indices, generated_texts):
                    prediction = self._extract_prediction(response)
                    results[idx] = (response, prediction)

                    # Cache the result
                    if self.config.use_cache or self.config.overwrite_cache:
                        question = questions[idx]
                        cache_key = f"rag:{question}|"
                        _inference_cache.set(self.config, cache_key, response)

            except Exception as e:
                error_msg = f"Batch RAG failed: {str(e)}"
                print(f"Error: {error_msg}")
                # Fill errors for uncached items
                for idx in uncached_indices:
                    results[idx] = (error_msg, "[ERROR]")

        return results

    def _extract_prediction(self, response: str, method: str = None) -> str:
        """Extract the final prediction from the model response."""
        response = response.strip()

        # Use regex to find all <answer>...</answer> patterns
        answer_matches = re.findall(r'<answer>(.*?)</answer>', response,
                                    re.DOTALL)

        if answer_matches:
            # For searchr1 and papersearchr1, use the second occurrence if available
            if method in ["searchr1", "papersearchr1"
                          ] and len(answer_matches) >= 3:
                prediction = answer_matches[2].strip()
            else:
                prediction = answer_matches[0].strip()

            if prediction:
                return prediction

        # Fallback patterns using regex
        fallback_patterns = [
            r'Final Answer:\s*(.*?)(?:\n|$)',
            r'Answer:\s*(.*?)(?:\n|$)',
        ]

        for pattern in fallback_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                prediction = match.group(1).strip()
                if prediction:
                    break
        else:
            # Take the first meaningful line
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and not re.match(
                        r'^(Question:|Context:|Instructions:|Let\'s think)',
                        line, re.IGNORECASE):
                    prediction = line
                    break
            else:
                # Fallback to first sentence
                sentences = response.split('.')
                prediction = sentences[0].strip() if sentences else response

        # Clean up prediction - remove common prefixes using regex
        prefix_pattern = r'^(the answer is |the final answer is |therefore,? |thus,? |in conclusion,? |so,? |hence,? )'
        prediction = re.sub(prefix_pattern,
                            '',
                            prediction,
                            flags=re.IGNORECASE).strip()

        # Keep only first line
        prediction = prediction.split('\n')[0].strip()

        return prediction


def llm_judge_batch(questions: List[str],
                    ground_truths: List[str],
                    predictions: List[str],
                    judge_model: str = "openai/gpt-4") -> List[Dict[str, Any]]:
    """
    Use an LLM to judge the quality of predictions in batch.
    
    Args:
        questions: List of original questions
        ground_truths: List of correct answers
        predictions: List of model predictions
        judge_model: The model to use for judging
    
    Returns:
        List of judgment dictionaries
    """
    # Create prompts for batch processing
    judge_prompts = []
    for question, ground_truth, prediction in zip(questions, ground_truths,
                                                  predictions):
        judge_prompt = f"""PPlease evaluate the following answer to a question.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {prediction}

Please provide:
1. A correctness score either 0 or 1. 
2. A brief explanation of your evaluation
3. Whether the prediction is "correct" or "incorrect"

The answer should be a single entity. 

- The answer is correct **only if it matches all key aspects, specificity, and detail of the ground truth, even if differently phrased (e.g., synonym or rewording that does not lose or broaden meaning).**
- If the prediction omits, generalizes, or adds information not present in the ground truth, it should be considered **incorrect**.
- **Even if the answer is a valid synonym, ensure it covers the same detail and scope.**
- For example, if the ground truth is "heart muscle," correct answers include "cardiac muscle" or "muscle of the heart," but NOT just "muscle" or "organ."
- **Any answer that is more vague, less specific, or encompasses a broader/narrower category than the ground truth should be marked as incorrect.**

Respond in JSON format:
{{
    "score": <score>,
    "explanation": "<explanation>",
    "judgment": "<correct/incorrect>"
}}"""
        judge_prompts.append(judge_prompt)

    try:
        # Use asyncio.run to call the async batch function from this sync context
        responses, _ = asyncio.run(
            call_llm_batch(prompts=judge_prompts,
                           model_name=judge_model,
                           max_tokens=500,
                           temperature=0.1,
                           json_mode=True,
                           max_concurrent=10))

        # Parse responses
        judgments = []
        for i, response in enumerate(responses):
            try:
                if response.startswith("[ERROR:"):
                    judgments.append({
                        "score": 0,
                        "explanation": f"Judge failed: {response}",
                        "judgment": "error"
                    })
                else:
                    judgment = json.loads(response)
                    judgments.append(judgment)
            except json.JSONDecodeError as e:
                print(f"Failed to parse judge response {i}: {e}")
                judgments.append({
                    "score": 0,
                    "explanation": f"JSON parse failed: {response[:100]}...",
                    "judgment": "error"
                })

        return judgments

    except Exception as e:
        print(f"LLM judge batch failed: {e}")
        # Return error judgments for all items
        return [{
            "score": 0,
            "explanation": f"Batch judge failed: {str(e)}",
            "judgment": "error"
        } for _ in questions]


def _extract_model_name(model_id: str) -> str:
    """Extract a clean model name from model_id for filename generation."""
    # Handle HuggingFace model IDs (e.g., "Qwen/Qwen2.5-3B-Instruct" -> "qwen25-3b-instruct")
    # Handle local paths (e.g., "/path/to/model" -> "model")

    if "/" in model_id:
        # Take the last part after the last slash
        model_name = model_id.split("/")[-1]
    else:
        model_name = model_id

    # Clean the model name for filename usage
    # Convert to lowercase and replace dots/spaces with hyphens
    model_name = model_name.lower()
    model_name = model_name.replace(".", "").replace(" ",
                                                     "-").replace("_", "-")

    return model_name


def _generate_output_path(config: InferenceConfig) -> str:
    """Generate standardized output path for all methods."""
    # Extract dataset name and replace "/" with "-"
    dataset_name = config.dataset_id.replace("/", "-")
    output_dir = f"results/eval/run_inference/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Extract clean model name for filename
    model_name = _extract_model_name(config.model_id)

    # Create filename with method and model name
    filename_parts = [config.method, model_name]

    # Add retriever type and corpus filename for RAG method
    if config.method == "rag":
        if config.rag_retriever_type:
            filename_parts.append(config.rag_retriever_type)
        if config.rag_corpus_filename:
            # Remove file extension from corpus filename
            corpus_name = os.path.splitext(config.rag_corpus_filename)[0]
            filename_parts.append(corpus_name)

    # Join with underscores
    filename = "_".join(filename_parts)
    output_path = f"{output_dir}/{filename}.csv"

    return output_path


def run_inference(config: InferenceConfig,
                  output_path: Optional[str] = None,
                  run_judge: bool = False) -> pd.DataFrame:
    """
    Run inference on the evaluation dataset.
    
    Args:
        config: Inference configuration
        output_path: Optional path to save results
        run_judge: Whether to run LLM judge evaluation
    
    Returns:
        DataFrame with results
    """
    print(f"Starting inference with method: {config.method}")
    print(f"Dataset: {config.dataset_id}")
    print(f"Model: {config.model_id}")
    print(f"Batch size: {config.batch_size}")
    print(f"First N examples: {config.first_n}")

    # Generate output path if not provided
    if output_path is None:
        output_path = _generate_output_path(config)
        print(f"Auto-generated output path: {output_path}")

    # Load dataset
    try:
        split = "test"
        dataset = load_dataset(config.dataset_id, split=split)
        df = dataset.to_pandas()
        print(f"Loaded {len(df)} examples from dataset")

        # Filter for cais/hle dataset to Biology subject only
        if config.dataset_id == "cais/hle" and 'raw_subject' in df.columns:
            before_filter = len(df)
            df = df[df['raw_subject'] == 'Biology']
            print(f"Filtered to Biology subject: {len(df)}/{before_filter} examples")

        # Limit to first N examples
        if config.first_n > 0 and len(df) > config.first_n:
            df = df.head(config.first_n)
            print(f"Limited to first {config.first_n} examples")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return pd.DataFrame()

    # Initialize inference engine
    engine = InferenceEngine(config)

    # Extract questions - dataset has 'question' and 'answer' columns, no context
    questions = []
    for _, row in df.iterrows():
        question = row.get('question', '')
        questions.append(question)

    # For most methods, contexts are empty since dataset doesn't have context
    # For RAG, contexts will be retrieved dynamically during inference
    contexts = [""] * len(questions)

    # Run inference based on method
    if config.method == "direct":
        print("Running batch direct inference...")
        all_results = engine.direct_inference_batch(questions, contexts)

    elif config.method == "cot":
        print("Running batch CoT inference...")
        all_results = engine.cot_inference_batch(questions, contexts)

    elif config.method == "rag":
        print("Running batch RAG inference...")
        all_results = engine.rag_inference_batch(questions, contexts)

    else:
        raise ValueError(f"Unknown method: {config.method}")

    # Create results DataFrame
    results_data = []
    for i, (response, prediction) in enumerate(all_results):
        result_row = df.iloc[i].to_dict()
        result_row['model_output'] = response
        result_row['prediction'] = prediction
        result_row['method'] = config.method
        result_row['model_id'] = config.model_id
        results_data.append(result_row)

    results_df = pd.DataFrame(results_data)

    print(f"Inference complete! Processed {len(results_df)} examples")

    # Always compute exact match scores (moved outside of run_judge check)
    if len(results_df) > 0:
        print("Computing exact match scores...")

        # Prepare data for exact match computation
        predictions = []
        golden_answers = []

        for idx, row in results_df.iterrows():
            predictions.append(row['prediction'])
            # Use golden_answers column if available, otherwise fallback to answer
            if 'golden_answers' in row:
                golden_answers.append(row['golden_answers'])
            else:
                golden_answers.append(row['answer'])

        # Compute exact match scores using golden_answers
        em_scores = []
        for pred, gold in zip(predictions, golden_answers):
            # Handle different answer formats based on dataset
            if isinstance(gold, list):
                # For datasets like PaperSearchQA where golden_answers is already a list
                clean_gold_list = {"target": gold}
            elif isinstance(gold, str):
                # For datasets like cais/hle where answer is a single string
                clean_gold_list = {"target": [gold]}
            else:
                # Fallback: convert to list (this was the problematic line for strings)
                clean_gold_list = {"target": [str(gold)]}

            # em_score func expects the answer to be in a very long answer trace with these answer tags:
            pred_str = f"<answer></answer><answer>{pred}</answer>"
            em_score = compute_score_em(pred_str, clean_gold_list)
            em_scores.append(em_score)

        results_df['em_score'] = em_scores

        # Calculate average EM score
        avg_em_score = np.mean(em_scores)
        print(f"Average Exact Match Score: {avg_em_score:.3f}")

    # Run LLM judge if requested
    if run_judge and len(results_df) > 0:
        print("Running LLM judge evaluation...")

        # Prepare data for batch processing (reuse from exact match computation above)
        questions = []
        ground_truths = []

        for idx, row in results_df.iterrows():
            questions.append(row['question'])
            ground_truths.append(row['answer'])

        # Run batch judgment
        # use the full 4o model bc it's for an eval set that isn't that big anyway
        judgments = llm_judge_batch(questions,
                                    ground_truths,
                                    predictions,
                                    judge_model="openai/gpt-4o")

        # Add judgment results to DataFrame
        for key in ['score', 'explanation', 'judgment']:
            results_df[f'judge_{key}'] = [j.get(key, None) for j in judgments]

        # Print judge summary statistics
        if 'judge_score' in results_df.columns:
            avg_judge_score = results_df['judge_score'].mean()
            print(f"Average LLM Judge Score: {avg_judge_score:.3f}")

    # Always save updated results and summary (moved outside of run_judge check)
    if output_path and len(results_df) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        # Print summary statistics to terminal
        print("\n" + "=" * 40)
        print("PaperSearchRL Evaluation Summary")
        print("=" * 40)
        print(f"Method: {config.method}")
        print(f"Model: {config.model_id}")
        print(f"Dataset: {config.dataset_id}")
        if config.method == "rag":
            print(f"Retriever Type: {config.rag_retriever_type}")
        print(f"Number of examples: {len(results_df)}")
        print("\nPerformance Metrics:")
        print("-" * 20)
        print(f"Average Exact Match Score: {avg_em_score:.3f}")
        if 'judge_score' in results_df.columns:
            avg_judge_score = results_df['judge_score'].mean()
            print(f"Average LLM Judge Score: {avg_judge_score:.3f}")
        print("=" * 40 + "\n")

        # Save summary statistics to txt file
        summary_path = output_path.replace('.csv', '.txt')
        with open(summary_path, 'w') as f:
            f.write("PaperSearchRL Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Method: {config.method}\n")
            f.write(f"Model: {config.model_id}\n")
            f.write(f"Dataset: {config.dataset_id}\n")
            if config.method == "rag":
                f.write(f"Retriever Type: {config.rag_retriever_type}\n")
            f.write(f"Number of examples: {len(results_df)}\n\n")
            f.write("Performance Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Exact Match Score: {avg_em_score:.3f}\n")
            if 'judge_score' in results_df.columns:
                avg_judge_score = results_df['judge_score'].mean()
                f.write(f"Average LLM Judge Score: {avg_judge_score:.3f}\n")

        print(f"Summary statistics saved to: {summary_path}")

    print("Evaluation complete!")

    return results_df


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Run inference evaluation")

    # Main arguments
    parser.add_argument("--method",
                        type=str,
                        required=True,
                        choices=["direct", "cot", "rag"],
                        help="Inference method to use")
    parser.add_argument(
        "--dataset_id",
        type=str,
        # default="jmhb/PaperSearchRL_v1_n10000_test500",
        default="jmhb/PaperSearchRL_v4_gv2_n3000_test500",
        help="HuggingFace dataset ID")
    parser.add_argument("--model_id",
                        type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model ID or path")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=
        "Path to save results CSV. If blank (recommended), results will be saved to ./results/eval/{dataset_name}/{method}_{model_name}.csv"
    )
    parser.add_argument("--first_n",
                        type=int,
                        default=10000,
                        help="Number of examples to process (default: 10000)")

    # Generation parameters
    parser.add_argument("--max_tokens",
                        type=int,
                        default=1024,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size for inference")

    # Method-specific parameters
    parser.add_argument("--rag_top_k",
                        type=int,
                        default=3,
                        help="Top-k documents for RAG")
    parser.add_argument("--retriever_type",
                        type=str,
                        choices=["e5", "bm25"],
                        default="bm25",
                        help="Retriever type for RAG method (e5 or bm25)")
    parser.add_argument("--corpus_filename",
                        type=str,
                        default="pubmed.jsonl",
                        help="Corpus filename for RAG method")

    # VLLM-specific parameters
    parser.add_argument("--vllm_gpu_memory_utilization",
                        type=float,
                        default=0.9,
                        help="GPU memory utilization for VLLM")
    parser.add_argument("--vllm_max_model_len",
                        type=int,
                        default=4096,
                        help="Maximum model length for VLLM")

    # Other options
    parser.add_argument("--no_cache",
                        action="store_true",
                        help="Disable caching")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help=
        "Skip reading from cache but write new results to cache (overwrite existing)"
    )
    parser.add_argument("--run_judge",
                        action="store_true",
                        help="Run LLM judge evaluation")

    args = parser.parse_args()

    # Create config
    config = InferenceConfig(
        method=args.method,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        first_n=args.first_n,
        rag_top_k=args.rag_top_k,
        rag_retriever_type=args.retriever_type,
        rag_corpus_filename=args.corpus_filename,
        use_cache=not args.no_cache,
        overwrite_cache=args.overwrite_cache,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len)

    # Run inference
    results_df = run_inference(config, args.output_path, args.run_judge)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
