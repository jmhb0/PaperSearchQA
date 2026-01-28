"""
Batch RAG evaluation.
This was vibecoded with claude 4, but has been verified. 

Called by run_inference.py functions. 
The SamplingParams below assume Qwen2.5 chat template.


"""
import transformers
import torch
import numpy as np
from datasets import load_dataset
import requests
import pandas as pd
from tqdm import tqdm
from qa_em import compute_score_em
import re
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams
import os
import ipdb
import sys

# Configuration
model_id = 'Qwen/Qwen2.5-3B-Instruct'


class BatchRAG:

    def __init__(self,
                 vllm_model: LLM,
                 tokenizer,
                 retrieval_topk: int = 3,
                 retriever_type: str = None,
                 corpus_filename: str = None,
                 retrieval_batch_size: int = 500):
        """Initialize BatchRAG with existing vLLM model and tokenizer."""
        self.vllm_model = vllm_model
        self.tokenizer = tokenizer
        self.retrieval_topk = retrieval_topk
        self.retriever_type = retriever_type
        self.corpus_filename = corpus_filename
        self.retrieval_batch_size = retrieval_batch_size

        # vLLM sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024,
            stop=["</answer>", "<|im_end|>", "</s>"])

        print("BatchRAG initialized successfully!")

    def batch_retrieve(self,
                       questions: List[str],
                       retriever_type: str = None,
                       corpus_filename: str = None,
                       batch_size: int = 500,
                       base_timeout: int = 60) -> List[str]:
        """Batch retrieve documents for all questions with chunking for large datasets."""
        print(f"Batch retrieving for {len(questions)} questions...")

        # Use instance retriever_type if parameter not provided
        if retriever_type is None:
            retriever_type = self.retriever_type

        # Use instance corpus_filename if parameter not provided
        if corpus_filename is None:
            corpus_filename = self.corpus_filename

        # Prepare all queries
        formatted_questions = []
        for question in questions:
            if not question.strip().endswith('?'):
                question = question.strip() + '?'
            formatted_questions.append(question)

        # Process in chunks if dataset is large
        all_results = []
        total_batches = (len(formatted_questions) + batch_size -
                         1) // batch_size

        for batch_idx in range(0, len(formatted_questions), batch_size):
            batch_questions = formatted_questions[batch_idx:batch_idx +
                                                  batch_size]
            batch_num = batch_idx // batch_size + 1

            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)..."
            )

            # Calculate timeout based on batch size (more time for larger batches)
            timeout = max(base_timeout,
                          len(batch_questions) *
                          0.2)  # 0.2 seconds per question minimum
            timeout = min(timeout, 300)  # Cap at 5 minutes

            # Single batch retrieval call
            payload = {
                "queries": batch_questions,
                "topk": self.retrieval_topk,
                "return_scores": True
            }

            response = requests.post("http://127.0.0.1:8001/retrieve",
                                     json=payload,
                                     timeout=timeout)
            response.raise_for_status()
            response_data = response.json()

            # Validate retriever type if specified (only check on first batch)
            if batch_idx == 0 and retriever_type is not None:
                actual_retriever_type = response_data.get('retriver_type')
                if actual_retriever_type != retriever_type:
                    raise ValueError(
                        f"Retriever type mismatch: requested '{retriever_type}' "
                        f"but got '{actual_retriever_type}' from server")

            # Validate corpus filename if specified (only check on first batch)
            if batch_idx == 0 and corpus_filename is not None:
                actual_corpus_filename = response_data.get('corpus_filename')
                if actual_corpus_filename != corpus_filename:
                    raise ValueError(
                        f"Corpus filename mismatch: requested '{corpus_filename}' "
                        f"but got '{actual_corpus_filename}' from server")

            batch_results = response_data['result']
            all_results.extend(batch_results)

            # except (requests.exceptions.ConnectionError,
            #         requests.exceptions.Timeout) as e:
            #     print(
            #         f"\n❌ ERROR: Cannot connect to retrieval server on localhost:8001"
            #     )
            #     print(f"Connection error: {e}")
            #     print(f"Failed on batch {batch_num}/{total_batches}")
            #     print(
            #         "Please start the retrieval server before running RAG method."
            #     )
            #     print("Example: python search_r1/search/retrieval_server.py")
            #     sys.exit(1)
            # except requests.exceptions.RequestException as e:
            #     print(f"\n❌ ERROR: HTTP request to retrieval server failed")
            #     print(f"Request error: {e}")
            #     print(f"Failed on batch {batch_num}/{total_batches}")
            #     print(
            #         "Please check the retrieval server status and configuration."
            #     )
            #     sys.exit(1)
            # except Exception as e:
            #     print(f"\n❌ ERROR: Unexpected error during batch retrieval")
            #     print(f"Error: {e}")
            #     print(f"Failed on batch {batch_num}/{total_batches}")
            #     sys.exit(1)

        # Format retrieved documents
        retrieved_contexts = []
        for question_results in all_results:
            context = self._passages2string(question_results)
            retrieved_contexts.append(context)

        return retrieved_contexts

    def _passages2string(self, retrieval_result: List[Dict]) -> str:
        """Convert retrieval results to formatted string."""
        if not retrieval_result:
            return "No relevant documents found."

        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference.strip()

    def prepare_rag_prompts(self, questions: List[str],
                            contexts: List[str]) -> List[str]:
        """Prepare RAG prompts with retrieved context."""
        prompts = []

        for question, context in zip(questions, contexts):
            if not question.strip().endswith('?'):
                question = question.strip() + '?'

            prompt = f"""Answer the given question using the provided context. If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Instructions: Provide a direct answer to the question between the tags <answer> and </answer>. Your answer should be a single entity, fact, or short phrase based on the context. Do not provide explanations or reasoning.

Answer:"""

            # Apply chat template if available
            if self.tokenizer.chat_template:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": prompt
                    }],
                    add_generation_prompt=True,
                    tokenize=False)
            else:
                formatted_prompt = prompt

            prompts.append(formatted_prompt)

        return prompts

    def batch_generate(self,
                       prompts: List[str],
                       sampling_params: SamplingParams = None) -> List[str]:
        """Generate answers for all prompts using vLLM batch inference."""
        print(f"Batch generating for {len(prompts)} prompts...")

        if sampling_params is None:
            sampling_params = self.sampling_params

        # Generate with vLLM
        outputs = self.vllm_model.generate(prompts, sampling_params)

        # Extract generated text
        generated_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)

        return generated_texts

    def run_batch_rag(
            self,
            questions: List[str],
            sampling_params: SamplingParams = None,
            retriever_type: str = None,
            corpus_filename: str = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """Run complete batch RAG pipeline and return contexts, responses, and prompts."""
        print(f"Running batch RAG for {len(questions)} questions...")

        # Step 1: Batch retrieval
        contexts = self.batch_retrieve(questions, retriever_type,
                                       corpus_filename)

        # Step 2: Prepare prompts
        prompts = self.prepare_rag_prompts(questions, contexts)
        # Step 3: Batch generation
        generated_texts = self.batch_generate(prompts, sampling_params)
        return contexts, generated_texts, prompts


# Standalone evaluation, more here for testing
def eval_batch_rag():
    """Evaluate batch RAG on the test dataset (standalone function)."""
    from vllm import LLM

    model_id = 'Qwen/Qwen2.5-3B-Instruct'
    checkpoint_path = "data/verl_checkpoints/20250604_grpo_bioasqv0_fullcorpus_qwenit_bm25/actor/global_step_100/"

    # Initialize vLLM model
    if checkpoint_path and os.path.exists(checkpoint_path):
        vllm_model = LLM(model=checkpoint_path,
                         dtype=torch.bfloat16,
                         gpu_memory_utilization=0.8)
    else:
        vllm_model = LLM(model=model_id,
                         dtype=torch.bfloat16,
                         gpu_memory_utilization=0.8)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Initialize BatchRAG
    batch_rag = BatchRAG(vllm_model, tokenizer, retrieval_topk=3)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("jmhb/bioasq_trainv0_n1609_test100", split="test")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} test examples")

    # Extract questions and golden answers
    questions = df['question'].tolist()
    golden_answers_list = df['golden_answers'].tolist()

    # Run batch RAG
    contexts, generated_texts, prompts = batch_rag.run_batch_rag(questions)

    # Evaluate results
    print("Evaluating results...")
    final_results = []

    for i, (question, context, generated_text, prompt) in enumerate(
            zip(questions, contexts, generated_texts, prompts)):
        golden_answers = golden_answers_list[i]

        # Extract answer
        extracted_answer = extract_answer(generated_text)

        # Compute correctness score
        ground_truth = {'target': golden_answers}
        score = compute_score_em(solution_str=generated_text,
                                 ground_truth=ground_truth,
                                 method='strict',
                                 format_score=0.0,
                                 score=1.0)

        # Add evaluation metrics
        result = {
            'question': question,
            'golden_answers': golden_answers,
            'retrieved_context': context,
            'generated_text': generated_text,
            'extracted_answer': extracted_answer,
            'correctness_score': score,
            'full_prompt': prompt
        }
        final_results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(final_results)

    # Calculate metrics
    mean_score = results_df['correctness_score'].mean()
    correct_count = (results_df['correctness_score'] > 0).sum()
    total_count = len(results_df)

    print(f"\n=== BATCH RAG EVALUATION RESULTS ===")
    print(f"Mean correctness score: {mean_score:.4f}")
    print(f"Correct answers: {correct_count}/{total_count}")
    print(f"Accuracy: {(correct_count/total_count)*100:.2f}%")

    # Save results
    os.makedirs("results/batch_rag", exist_ok=True)
    output_path = "results/batch_rag/results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    return results_df


def extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        # Fallback: return first line of response
        return text.split('\n')[0].strip()


if __name__ == "__main__":
    # Run standalone evaluation
    results_df = eval_batch_rag()
