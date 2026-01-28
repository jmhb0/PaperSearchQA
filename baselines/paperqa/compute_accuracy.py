#!/usr/bin/env python3
"""
Compute accuracy from PaperQA evaluation results.

Usage:
    python eval/compute_paperqa_accuracy.py
    python eval/compute_paperqa_accuracy.py --results_file path/to/results.json
"""

import json
import re
import argparse
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from verl.utils.reward_score.qa_em import compute_score_em, em_check


def clean_answer_preview(answer_preview: str) -> str:
    """
    Clean answer_preview by removing unwanted keywords and patterns.

    Removes:
    - "References" sections
    - Patterns like "(chunk-2461406)"
    - Patterns like "Document 2461406"
    """
    if not answer_preview:
        return ""

    # Convert to string if not already
    text = str(answer_preview).strip()

    # Remove "References" section and everything after it (case insensitive)
    text = re.sub(r'\n\s*references.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\s*references.*$', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove patterns like "(chunk-123456)" for any number
    text = re.sub(r'\(chunk-\d+\)', '', text)

    # Remove patterns like "Document 123456" for any number
    text = re.sub(r'Document\s+\d+', '', text)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_ground_truth(dataset_name: str = "PaperSearchQA/PaperSearchQA", split: str = "test"):
    """Load ground truth answers from the dataset."""
    print(f"Loading ground truth from {dataset_name} ({split} split)...")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except ValueError:
        # If specified split doesn't exist, try train split
        print(f"Split '{split}' not found, trying 'train' split...")
        dataset = load_dataset(dataset_name, split="train")
    return dataset


def compute_accuracy(results_file: str = "paperqa_PaperSearchQA_BioASQ_factoid_results.json",
                    dataset_name: str = "PaperSearchQA/BioASQ_factoid"):
    """Compute accuracy by comparing results with ground truth."""

    # Load results
    print(f"Loading results from {results_file}...")
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Load ground truth dataset
    ground_truth_dataset = load_ground_truth(dataset_name)

    print(f"Loaded {len(results)} results and {len(ground_truth_dataset)} ground truth examples")

    # Process each result
    accuracy_scores = []
    detailed_results = []

    for result in results:
        if not result.get("success", False):
            print(f"Skipping failed result: {result['question_id']}")
            continue

        # Extract question ID and convert to dataset index (0-based)
        question_id = result["question_id"]  # e.g., "q_001"
        try:
            # Extract number from question_id (e.g., "q_001" -> 1, then convert to 0-based index)
            question_num = int(question_id.split('_')[1])
            dataset_index = question_num - 1  # Convert to 0-based index
        except (ValueError, IndexError):
            print(f"Warning: Could not parse question_id '{question_id}', skipping")
            continue

        if dataset_index >= len(ground_truth_dataset):
            print(f"Warning: Dataset index {dataset_index} out of range, skipping")
            continue

        # Get ground truth for this question
        ground_truth_row = ground_truth_dataset[dataset_index]

        # Handle different dataset formats
        if 'golden_answers' in ground_truth_row:
            golden_answers = ground_truth_row['golden_answers']
        elif 'answer' in ground_truth_row:
            # BioASQ format - answer is typically a list or string
            answer = ground_truth_row['answer']
            if isinstance(answer, list):
                golden_answers = answer
            else:
                golden_answers = [str(answer)]
        else:
            print(f"Warning: No answer field found for {question_id}, skipping")
            continue

        # Clean the predicted answer - check multiple possible fields
        raw_answer = result.get("answer_preview") or result.get("extracted_answer", "")

        # Debug: show what fields are available for first few results
        if len(detailed_results) < 3:
            available_fields = list(result.keys())
            print(f"DEBUG {question_id}: Available fields: {available_fields}")
            print(f"DEBUG {question_id}: answer_preview = {repr(result.get('answer_preview'))}")
            print(f"DEBUG {question_id}: extracted_answer = {repr(result.get('extracted_answer'))}")
            print(f"DEBUG {question_id}: full_response preview = {repr(result.get('full_response', '')[:100])}")

        cleaned_answer = clean_answer_preview(raw_answer)

        # Format for compute_score_em function
        ground_truth_dict = {'target': golden_answers}

        # Compute EM score directly using em_check since we already have extracted answer
        if cleaned_answer:
            score = em_check(cleaned_answer, golden_answers)
        else:
            score = 0

        accuracy_scores.append(score)

        # Store detailed result for debugging
        detailed_result = {
            'question_id': question_id,
            'dataset_index': dataset_index,
            'question': result.get('question', ''),
            'golden_answers': golden_answers,
            'raw_answer': raw_answer,
            'cleaned_answer': cleaned_answer,
            'score': score,
            'processing_time': result.get('processing_time', 0),
            'cached': result.get('cached', False)
        }
        detailed_results.append(detailed_result)

        # Print progress for first 15 and any correct answers
        if len(detailed_results) <= 15 or score > 0:
            correct_marker = "✓" if score > 0 else "✗"
            cached_marker = "💾" if detailed_result['cached'] else "🔥"
            print(f"{correct_marker} {question_id} {cached_marker} Score: {score:.1f}")
            print(f"   Question: {detailed_result['question'][:80]}...")
            print(f"   Golden: {golden_answers}")
            print(f"   Predicted: '{cleaned_answer}'")
            if raw_answer != cleaned_answer:
                print(f"   Raw: '{raw_answer}'")
            print()

    # Calculate final statistics
    if accuracy_scores:
        mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        num_correct = sum(1 for score in accuracy_scores if score > 0)
        total_questions = len(accuracy_scores)

        print("=" * 60)
        print("ACCURACY RESULTS")
        print("=" * 60)
        print(f"Total questions processed: {total_questions}")
        print(f"Correct answers: {num_correct}")
        print(f"Accuracy: {num_correct}/{total_questions} = {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")

        # Show cache statistics
        cached_count = sum(1 for r in detailed_results if r['cached'])
        print(f"Cache hits: {cached_count}/{total_questions} ({cached_count/total_questions*100:.1f}%)")

        # Show timing statistics
        processing_times = [r['processing_time'] for r in detailed_results]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            total_time = sum(processing_times)
            print(f"Average processing time: {avg_time:.2f}s")
            print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

        # Create paperqa/results directory if it doesn't exist
        results_dir = Path('paperqa/results')
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results for debugging
        results_filename = Path(results_file).name
        debug_file = results_dir / results_filename.replace('.json', '_accuracy_debug.json')
        with open(debug_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"\nDetailed results saved to: {debug_file}")

        # Save results as CSV for easier analysis
        csv_file = results_dir / results_filename.replace('.json', '_accuracy_results.csv')
        df = pd.DataFrame(detailed_results)
        df.to_csv(csv_file, index=False)
        print(f"CSV results saved to: {csv_file}")

        # Compute and save per-category accuracy if category mapping exists
        try:
            compute_per_category_accuracy(detailed_results, results_dir, results_filename)
        except ValueError as e:
            print(f"\n{e}")
            print("Per-category analysis skipped. Continuing with overall accuracy results only.")

        return mean_accuracy, detailed_results
    else:
        print("No successful results found!")
        return 0.0, []


def load_category_mapping():
    """
    Load category mapping from SearchR1 evaluation CSV files if available.
    This tries to create a mapping from questions to categories.
    """
    category_mapping = {}

    # Try to find SearchR1 CSV files with category information
    search_paths = [
        "results/eval/run_inference/jmhb-papersearchr1/*.csv",
        "results/jmhb-papersearchr1/*per_category.csv"
    ]

    try:
        import glob
        for search_path in search_paths:
            csv_files = glob.glob(search_path)
            if csv_files:
                # Try to load one of the CSV files to get question-category mapping
                df = pd.read_csv(csv_files[0])
                if 'question' in df.columns and 'cat' in df.columns:
                    for _, row in df.iterrows():
                        category_mapping[row['question']] = row['cat']
                    print(f"Loaded category mapping from {csv_files[0]} with {len(category_mapping)} entries")
                    break
    except Exception as e:
        print(f"Could not load category mapping: {e}")

    return category_mapping


def compute_per_category_accuracy(detailed_results, results_dir, results_filename):
    """Compute per-category accuracy if category mapping is available."""

    # Try to load category mapping
    category_mapping = load_category_mapping()

    if not category_mapping:
        raise ValueError("ERROR: No category mapping available. The BioASQ dataset does not contain explicit category information. Per-category analysis requires a dataset with category annotations (like the PaperSearchR1 evaluation data).")

    # Add categories to detailed results
    categorized_results = []
    unknown_count = 0
    for result in detailed_results:
        question = result['question']
        category = category_mapping.get(question, 'Unknown')
        if category == 'Unknown':
            unknown_count += 1
        result_with_cat = result.copy()
        result_with_cat['category'] = category
        categorized_results.append(result_with_cat)

    # Check if too many questions are unmapped
    total_questions = len(detailed_results)
    unknown_percentage = unknown_count / total_questions

    if unknown_percentage > 0.5:  # More than 50% unmapped
        raise ValueError(f"ERROR: Category mapping failed - {unknown_count}/{total_questions} ({unknown_percentage:.1%}) questions could not be mapped to categories. Questions may not match between datasets.")

    if unknown_count > 0:
        print(f"WARNING: {unknown_count}/{total_questions} ({unknown_count/total_questions:.1%}) questions could not be mapped to categories")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(categorized_results)

    # Compute per-category statistics
    category_stats = df.groupby('category')['score'].agg([
        'count',  # total questions
        'sum',    # correct answers
        'mean'    # accuracy
    ]).round(4)

    category_stats.columns = ['total_questions', 'correct_answers', 'accuracy']
    category_stats = category_stats.sort_values('accuracy', ascending=False)

    # Save per-category results
    per_cat_file = results_dir / results_filename.replace('.json', '_per_category.csv')
    category_stats.to_csv(per_cat_file)

    # Create summary
    summary_file = results_dir / results_filename.replace('.json', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Per-Category Accuracy Summary\n")
        f.write("=" * 50 + "\n\n")

        overall_accuracy = df['score'].mean()
        total_correct = int(df['score'].sum())
        total_questions = len(df)

        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_questions})\n\n")
        f.write("Category-wise Results:\n")
        f.write("-" * 30 + "\n")

        for cat, row in category_stats.iterrows():
            f.write(f"{cat:<35} {row['accuracy']:.4f} ({int(row['correct_answers'])}/{int(row['total_questions'])})\n")

    print(f"Per-category results saved to: {per_cat_file}")
    print(f"Summary saved to: {summary_file}")

    # Print category results to console
    print("\nPer-category accuracy:")
    for cat, row in category_stats.head(10).iterrows():
        print(f"  {cat:<35} {row['accuracy']:.4f} ({int(row['correct_answers'])}/{int(row['total_questions'])})")


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy from PaperQA evaluation results")
    parser.add_argument("--results_file", type=str, default="paperqa_PaperSearchQA_BioASQ_factoid_results.json",
                       help="Path to PaperQA results JSON file")
    parser.add_argument("--dataset_name", type=str, default="PaperSearchQA/BioASQ_factoid",
                       help="Dataset name for ground truth")

    args = parser.parse_args()

    accuracy, detailed_results = compute_accuracy(args.results_file, args.dataset_name)

    return accuracy


if __name__ == "__main__":
    main()