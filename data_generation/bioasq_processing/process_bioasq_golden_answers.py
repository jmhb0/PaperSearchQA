#!/usr/bin/env python3
"""
Process BioASQ dataset to add golden_answers (synonym lists).

This script processes the BioASQ-taskb dataset and adds golden_answers
(synonym lists) for each answer using GPT-4o-mini. This was used for creating
the BioASQ test set used in the paper.

IMPORTANT - DATA SOURCE REQUIREMENTS:
    This script requires access to the BioASQ-taskb dataset, which is NOT
    publicly available. To use this script, you must:

    1. Register at http://bioasq.org/
    2. Download the BioASQ Task B data
    3. Upload it to HuggingFace as a private dataset
    4. Update the load_dataset() calls below to point to your dataset

    This script is provided AS-IS for documentation purposes to show how
    the golden_answers were generated for the BioASQ test set used in the paper.

    The processed BioASQ test set (with golden_answers) is available at:
    https://huggingface.co/datasets/PaperSearchQA/BioASQ_factoid

Usage:
    python process_bioasq_golden_answers.py --n_samples 10000 --n_test 100
    python process_bioasq_golden_answers.py --factoid_yesno --n_samples 10000 --n_test 100 --pcnt 0.2
"""

import argparse
import ast
import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import os
from typing import List, Dict, Any, Union
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_pipeline'))
from api import call_llm_batch
import ipdb
import numpy as np

# Configuration for HuggingFace uploads
# Set ENABLE_HF_UPLOAD=1 to explicitly enable uploads (requires HF_USERNAME and HF_TOKEN)
ENABLE_HF_UPLOAD = os.environ.get("ENABLE_HF_UPLOAD", "0") == "1"
HF_USERNAME = os.environ.get("HF_USERNAME", "your_username")  # Used for dataset naming
OUTPUT_DIR = "output"


def parse_answer_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'answer' column from string representation to actual lists.
    
    Args:
        df: DataFrame with 'answer' column containing string representations of lists
        
    Returns:
        DataFrame with 'answer' column converted to actual lists
    """

    def safe_literal_eval(x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # If literal_eval fails, try to handle as a simple string
            return [x] if isinstance(x, str) else x

    df['answer'] = df['answer'].apply(safe_literal_eval)
    return df


def generate_synonyms_prompt(answers: List[str]) -> str:
    """
    Generate the prompt for synonym generation.
    
    Args:
        answers: List of answer strings
        
    Returns:
        Formatted prompt string
    """
    answers_str = str(answers)

    prompt = f"""I am generating a dataset for biological question answering.

They are questions that have simple factoid answers - meaning the answer is a single entity.

Below I'll give the target answer or answers.

However the same entity might have synonyms that are equivalent.

Your task is to return a python list of synonyms for each unique entity mentioned in the answers.

For example, if the target answer was "c-Jun NH2-terminal kinase", then you should return a list of synonyms ["JNK", "c-Jun N-terminal kinase", "c-Jun amino-terminal kinase", "c-Jun NH2-terminal kinase"]

Return a python list of all the common synonyms. Return as many as you can think of. Include the original answer(s) in the list as well.

You must respond with valid JSON containing only the list of synonyms.

ANSWERS
{answers_str}"""

    return prompt


def process_gpt_response(response: str) -> List[str]:
    """
    Process GPT response to extract the python list of synonyms.
    
    Args:
        response: Raw response from GPT
        
    Returns:
        List of synonym strings
    """
    try:
        response = response.strip()

        # First try to parse as JSON (since we're using json_mode)
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and 'synonyms' in parsed:
                # Handle cases where GPT wraps the list in an object
                return parsed['synonyms']
            elif isinstance(parsed, dict):
                # Try to find any list value in the JSON object
                for value in parsed.values():
                    if isinstance(value, list):
                        return value
        except json.JSONDecodeError:
            pass

        # Fallback: try to parse as Python literal (ast.literal_eval)
        try:
            if response.startswith('[') and response.endswith(']'):
                return ast.literal_eval(response)

            # Try to find a list within the response
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    return ast.literal_eval(line)
        except (ValueError, SyntaxError):
            pass

        # Final fallback: return the response as a single-item list
        print(f"Warning: Could not parse response as list: {response}")
        return [response]

    except Exception as e:
        print(f"Error processing GPT response: {e}")
        print(f"Response was: {response}")
        return ["[ERROR_PARSING_RESPONSE]"]


def generate_golden_answers(df: pd.DataFrame,
                            batch_size: int = 50) -> pd.DataFrame:
    """
    Generate golden_answers column using GPT-4o mini in batch mode.
    
    Args:
        df: DataFrame with 'answer' column
        batch_size: Batch size for API calls
        
    Returns:
        DataFrame with new 'golden_answers' column.
    """
    print(f"Generating synonyms for {len(df)} samples...")

    # Prepare prompts for all rows
    prompts = []
    for _, row in df.iterrows():
        prompt = generate_synonyms_prompt(row['answer'])
        prompts.append(prompt)

    print(f"Making batch API calls with {len(prompts)} prompts...")

    # Call GPT-4o mini in batch mode
    responses, costs = call_llm_batch(
        prompts=prompts,
        model_name="openai/gpt-4o-mini",
        max_tokens=500,
        temperature=0.1,  # Low temperature for more consistent outputs
        max_concurrent=10,  # Conservative concurrent limit
        json_mode=True  # Enable JSON mode for structured output
    )

    print("Processing responses...")

    # Process responses to extract synonym lists
    golden_answers = []
    for i, response in enumerate(responses):
        if response.startswith('[ERROR:'):
            print(f"API error for row {i}: {response}")
            # Fallback to original answers
            golden_answers.append(df.iloc[i]['answer'])
        else:
            synonyms = process_gpt_response(response)
            golden_answers.append(synonyms)

    # Add the golden_answers column right after the answer column
    df = df.copy()
    answer_col_idx = df.columns.get_loc('answer')
    df.insert(answer_col_idx + 1, 'golden_answers', golden_answers)

    return df


def save_or_upload_dataset(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                           dataset_name_suffix: str) -> None:
    """
    Save dataset locally and optionally upload to HuggingFace.

    By default, saves to local parquet files in OUTPUT_DIR.
    If ENABLE_HF_UPLOAD=1, also uploads to HuggingFace.

    Args:
        data: Either a DataFrame for single split or dict of DataFrames for multiple splits
        dataset_name_suffix: Suffix for dataset name (e.g., "bioasq_trainv0_n1609")
                            Will be prefixed with HF_USERNAME
    """
    # Build dataset name with username prefix
    dataset_name_with_prefix = f"{HF_USERNAME}/{dataset_name_suffix}"

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Always save locally with username prefix in filename
    if isinstance(data, dict):
        # Multiple splits
        for split_name, df in data.items():
            output_file = os.path.join(OUTPUT_DIR, f"{dataset_name_with_prefix.replace('/', '_')}_{split_name}.parquet")
            df.to_parquet(output_file, index=False)
            print(f"✅ Saved {split_name} split to: {output_file}")
    else:
        # Single split
        output_file = os.path.join(OUTPUT_DIR, f"{dataset_name_with_prefix.replace('/', '_')}.parquet")
        data.to_parquet(output_file, index=False)
        print(f"✅ Saved dataset to: {output_file}")

    # Optionally upload to HuggingFace if explicitly enabled
    if ENABLE_HF_UPLOAD:
        print(f"\nUploading to HuggingFace as '{dataset_name_with_prefix}'...")

        if isinstance(data, dict):
            # Multiple splits
            dataset_dict = {}
            for split_name, df in data.items():
                dataset_dict[split_name] = Dataset.from_pandas(df)
            dataset = DatasetDict(dataset_dict)
        else:
            # Single DataFrame
            dataset = Dataset.from_pandas(data)

        # Upload to HuggingFace Hub
        dataset.push_to_hub(
            dataset_name_with_prefix,
            private=False,
            token=os.getenv("HF_TOKEN")
        )
        print(f"✅ Uploaded to: https://huggingface.co/datasets/{dataset_name_with_prefix}")
    else:
        print("\nℹ️  To upload to HuggingFace, set ENABLE_HF_UPLOAD=1:")
        print("   export ENABLE_HF_UPLOAD=1")
        print("   export HF_USERNAME=your_username")
        print("   export HF_TOKEN=your_token")


def build_bioasq_dataset(n_samples: int,
                         n_test: int = None,
                         batch_size: int = 50,
                         is_yesno: bool = False):
    """
    Build BioASQ dataset for either factoid or yes/no questions.
    
    Args:
        n_samples: Number of samples to process
        n_test: Number of samples to use as test set (default: None)
        batch_size: Batch size for API calls (default: 50)
        is_yesno: If True, filter for yes/no questions; if False, filter for factoid questions
    """
    # Set random seed for reproducibility
    np.random.seed(0)

    question_type = "yesno" if is_yesno else "factoid"

    print(f"Building training dataset with {n_samples} samples...")
    if n_test is not None:
        print(
            f"Will split into {n_samples - n_test} train and {n_test} test samples"
        )

    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("jmhb/BioASQ-taskb")

    # Convert to DataFrame
    df = dataset['train'].to_pandas()

    # Drop the unwanted index column if it exists
    if '__index_level_0__' in df.columns:
        df = df.drop(columns=['__index_level_0__'])

    print(f"Loaded {len(df)} total samples")

    # Filter for the specified question type
    print(f"Filtering for {question_type} questions...")
    df_filtered = df[df['type'] == question_type].copy()
    print(f"Found {len(df_filtered)} {question_type} questions")

    if len(df_filtered) < n_samples:
        print(
            f"Warning: Only {len(df_filtered)} {question_type} questions available, "
            f"but {n_samples} requested")
        n_samples = len(df_filtered)

    # Validate n_test if provided
    if n_test is not None:
        if n_test >= n_samples:
            raise ValueError(
                f"n_test ({n_test}) must be less than n_samples ({n_samples})")

    # Parse answer column
    print("Parsing answer column...")
    df_filtered = parse_answer_column(df_filtered)

    # Randomly sample n_samples from filtered questions
    print(
        f"Randomly sampling {n_samples} from {len(df_filtered)} {question_type} questions..."
    )
    df_subset = df_filtered.sample(n=n_samples, random_state=0).copy()
    print(f"Processing {len(df_subset)} samples")

    # Generate golden_answers
    if is_yesno:
        # For yes/no questions, golden_answers are just the parsed answers
        print(
            "For yes/no questions, using parsed answers directly as golden_answers..."
        )
        df_processed = df_subset.copy()
        # Add the golden_answers column right after the answer column
        answer_col_idx = df_processed.columns.get_loc('answer')
        df_processed.insert(answer_col_idx + 1, 'golden_answers',
                            df_processed['answer'])
    else:
        # For factoid questions, use GPT-4o mini to generate synonyms
        df_processed = generate_golden_answers(df_subset,
                                               batch_size=batch_size)

    # put the 'answer' column back to a string type bc the types are not consistent and it breaks dataset conversion
    df_processed['answer'] = [str(a) for a in df_processed['answer']]

    # Split into train/test if n_test is provided
    if n_test is not None:
        print(f"Splitting dataset into train/test sets...")

        # Reset index to ensure we have consecutive indices for sampling
        df_processed = df_processed.reset_index(drop=True)

        # Randomly sample test indices
        test_indices = np.random.choice(len(df_processed),
                                        size=n_test,
                                        replace=False)

        # Create boolean mask for test samples
        test_mask = np.zeros(len(df_processed), dtype=bool)
        test_mask[test_indices] = True

        df_test = df_processed[test_mask].copy()
        df_train = df_processed[~test_mask].copy()

        print(f"Train set: {len(df_train)} samples")
        print(f"Test set: {len(df_test)} samples")

        # Validate that we have the expected number of samples
        assert len(
            df_test
        ) == n_test, f"Expected {n_test} test samples, got {len(df_test)}"
        assert len(
            df_train
        ) == n_samples - n_test, f"Expected {n_samples - n_test} train samples, got {len(df_train)}"

        # Create dataset name and upload with splits
        if is_yesno:
            dataset_name = f"bioasq_yesno_trainv0_n{n_samples}_test{n_test}"
        else:
            dataset_name = f"bioasq_trainv0_n{n_samples}_test{n_test}"

        # Upload as single dataset with train/test splits
        save_or_upload_dataset({
            'train': df_train,
            'test': df_test
        }, dataset_name)

        print("✅ Dataset processing complete!")
        print(f"Dataset uploaded as: {dataset_name}")
        print(f"  - Train split: {len(df_train)} samples")
        print(f"  - Test split: {len(df_test)} samples")

        # Print sample of results for both sets
        print("\nSample train results:")
        for i in range(min(2, len(df_train))):
            row = df_train.iloc[i]
            print(f"\nTrain Sample {i+1}:")
            print(f"Question: {row['question'][:100]}...")
            print(f"Original answers: {row['answer']}")
            print(f"Golden answers: {row['golden_answers']}")

        print("\nSample test results:")
        for i in range(min(2, len(df_test))):
            row = df_test.iloc[i]
            print(f"\nTest Sample {i+1}:")
            print(f"Question: {row['question'][:100]}...")
            print(f"Original answers: {row['answer']}")
            print(f"Golden answers: {row['golden_answers']}")
    else:
        # Original behavior - single dataset
        if is_yesno:
            dataset_name = f"bioasq_yesno_trainv0_n{n_samples}"
        else:
            dataset_name = f"bioasq_trainv0_n{n_samples}"

        # Upload to HuggingFace
        save_or_upload_dataset(df_processed, dataset_name)

        print("✅ Dataset processing complete!")
        print(f"Dataset uploaded as: {dataset_name}")

        # Print sample of results
        print("\nSample results:")
        for i in range(min(3, len(df_processed))):
            row = df_processed.iloc[i]
            print(f"\nSample {i+1}:")
            print(f"Question: {row['question'][:100]}...")
            print(f"Original answers: {row['answer']}")
            print(f"Golden answers: {row['golden_answers']}")


def build_bioasq_factoid_yesno_dataset(n_samples: int,
                                       n_test: int = None,
                                       batch_size: int = 50,
                                       pcnt: float = 0.2):
    """
    Build BioASQ dataset containing both factoid and yes/no questions.
    
    Args:
        n_samples: Number of factoid samples to process for training
        n_test: Number of factoid samples to use as test set (default: None)
        batch_size: Batch size for API calls (default: 50)
        pcnt: Percentage composition of yesno questions in final train set (default: 0.2)
    """
    # Set random seed for reproducibility
    np.random.seed(0)

    print(f"Building combined factoid+yesno dataset...")
    print(
        f"Factoid: {n_samples} samples (train: {n_samples - (n_test or 0)}, test: {n_test or 0})"
    )

    # Calculate yesno sizes
    factoid_train_size = n_samples - (n_test or 0) if n_test else n_samples

    # Calculate yesno train size so that pcnt represents final composition
    # yesno_train_size / (factoid_train_size + yesno_train_size) = pcnt
    # Solving: yesno_train_size = (pcnt * factoid_train_size) / (1 - pcnt)
    yesno_train_size = int((pcnt * factoid_train_size) / (1 - pcnt))
    yesno_test_size = 100

    final_train_size = factoid_train_size + yesno_train_size
    actual_pcnt = yesno_train_size / final_train_size

    print(f"Yesno: train={yesno_train_size}, test={yesno_test_size}")
    print(
        f"Final train composition: {factoid_train_size} factoid + {yesno_train_size} yesno = {final_train_size} total"
    )
    print(f"Yesno percentage in final train set: {actual_pcnt:.1%}")

    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("jmhb/BioASQ-taskb")
    df = dataset['train'].to_pandas()

    # Drop the unwanted index column if it exists
    if '__index_level_0__' in df.columns:
        df = df.drop(columns=['__index_level_0__'])

    print(f"Loaded {len(df)} total samples")

    # Filter for factoid questions
    print("Processing factoid questions...")
    df_factoid = df[df['type'] == 'factoid'].copy()
    print(f"Found {len(df_factoid)} factoid questions")

    if len(df_factoid) < n_samples:
        print(
            f"Warning: Only {len(df_factoid)} factoid questions available, but {n_samples} requested"
        )
        n_samples = len(df_factoid)
        # Recalculate yesno train size
        factoid_train_size = n_samples - (n_test or 0) if n_test else n_samples
        yesno_train_size = int((pcnt * factoid_train_size) / (1 - pcnt))
        final_train_size = factoid_train_size + yesno_train_size
        actual_pcnt = yesno_train_size / final_train_size
        print(
            f"Recalculated: Yesno train={yesno_train_size}, final composition: {actual_pcnt:.1%}"
        )

    # Parse answer column for factoid
    print("Parsing factoid answer column...")
    df_factoid = parse_answer_column(df_factoid)

    # Sample factoid questions
    print(f"Randomly sampling {n_samples} factoid questions...")
    df_factoid_subset = df_factoid.sample(n=n_samples, random_state=0).copy()

    # Generate golden_answers for factoid questions
    df_factoid_processed = generate_golden_answers(df_factoid_subset,
                                                   batch_size=batch_size)

    # Add data_source column for factoid
    df_factoid_processed['data_source'] = 'bioasq_factoid'

    # Filter for yesno questions
    print("Processing yesno questions...")
    df_yesno = df[df['type'] == 'yesno'].copy()
    print(f"Found {len(df_yesno)} yesno questions")

    total_yesno_needed = yesno_train_size + yesno_test_size
    if len(df_yesno) < total_yesno_needed:
        print(
            f"Warning: Only {len(df_yesno)} yesno questions available, but {total_yesno_needed} needed"
        )
        # Adjust sizes proportionally
        ratio = len(df_yesno) / total_yesno_needed
        yesno_train_size = int(yesno_train_size * ratio)
        yesno_test_size = len(df_yesno) - yesno_train_size
        print(
            f"Adjusted: Yesno train={yesno_train_size}, test={yesno_test_size}"
        )

    # Parse answer column for yesno
    print("Parsing yesno answer column...")
    df_yesno = parse_answer_column(df_yesno)

    # Sample yesno questions
    total_yesno_samples = yesno_train_size + yesno_test_size
    print(f"Randomly sampling {total_yesno_samples} yesno questions...")
    df_yesno_subset = df_yesno.sample(n=total_yesno_samples,
                                      random_state=1).copy()

    # For yesno questions, golden_answers are just the parsed answers
    print("Setting golden_answers for yesno questions...")
    answer_col_idx = df_yesno_subset.columns.get_loc('answer')
    df_yesno_subset.insert(answer_col_idx + 1, 'golden_answers',
                           df_yesno_subset['answer'])

    # Add data_source column for yesno
    df_yesno_subset['data_source'] = 'bioasq_yesno'

    # Convert answer columns to string for both datasets
    df_factoid_processed['answer'] = [
        str(a) for a in df_factoid_processed['answer']
    ]
    df_yesno_subset['answer'] = [str(a) for a in df_yesno_subset['answer']]

    # Split data appropriately
    if n_test is not None:
        print("Splitting datasets...")

        # Reset indices
        df_factoid_processed = df_factoid_processed.reset_index(drop=True)
        df_yesno_subset = df_yesno_subset.reset_index(drop=True)

        # Split factoid data
        factoid_test_indices = np.random.choice(len(df_factoid_processed),
                                                size=n_test,
                                                replace=False)
        factoid_test_mask = np.zeros(len(df_factoid_processed), dtype=bool)
        factoid_test_mask[factoid_test_indices] = True

        df_factoid_test = df_factoid_processed[factoid_test_mask].copy()
        df_factoid_train = df_factoid_processed[~factoid_test_mask].copy()

        # Split yesno data
        yesno_test_indices = np.random.choice(len(df_yesno_subset),
                                              size=yesno_test_size,
                                              replace=False)
        yesno_test_mask = np.zeros(len(df_yesno_subset), dtype=bool)
        yesno_test_mask[yesno_test_indices] = True

        df_yesno_test = df_yesno_subset[yesno_test_mask].copy()
        df_yesno_train = df_yesno_subset[~yesno_test_mask].copy()

        # Combine train and test sets
        df_train_combined = pd.concat([df_factoid_train, df_yesno_train],
                                      ignore_index=True)
        df_test_combined = pd.concat([df_factoid_test, df_yesno_test],
                                     ignore_index=True)

        # Shuffle the combined datasets
        print("Shuffling combined datasets...")
        df_train_combined = df_train_combined.sample(
            frac=1, random_state=2).reset_index(drop=True)
        df_test_combined = df_test_combined.sample(
            frac=1, random_state=3).reset_index(drop=True)

        # Verify final composition
        train_factoid_count = (
            df_train_combined['data_source'] == 'bioasq_factoid').sum()
        train_yesno_count = (
            df_train_combined['data_source'] == 'bioasq_yesno').sum()
        final_composition = train_yesno_count / len(df_train_combined)

        print(f"Combined train set: {len(df_train_combined)} samples")
        print(
            f"  - Factoid: {train_factoid_count} samples ({train_factoid_count/len(df_train_combined):.1%})"
        )
        print(
            f"  - Yesno: {train_yesno_count} samples ({final_composition:.1%})"
        )
        print(f"Combined test set: {len(df_test_combined)} samples")
        print(f"  - Factoid: {len(df_factoid_test)} samples")
        print(f"  - Yesno: {len(df_yesno_test)} samples")

        # Create dataset name
        pcnt_int = int(pcnt * 100)
        dataset_name = f"bioasq_trainv0_factoidyesno_pcnt{pcnt_int}_n{n_samples}_test{n_test}"

        # Upload as single dataset with train/test splits
        save_or_upload_dataset(
            {
                'train': df_train_combined,
                'test': df_test_combined
            }, dataset_name)

        print("✅ Dataset processing complete!")
        print(f"Dataset uploaded as: {dataset_name}")
        print(f"  - Train split: {len(df_train_combined)} samples")
        print(f"  - Test split: {len(df_test_combined)} samples")

        # Print sample of results
        print("\nSample train results:")
        for i in range(min(2, len(df_train_combined))):
            row = df_train_combined.iloc[i]
            print(f"\nTrain Sample {i+1} ({row['data_source']}):")
            print(f"Question: {row['question'][:100]}...")
            print(f"Original answers: {row['answer']}")
            print(f"Golden answers: {row['golden_answers']}")

        print("\nSample test results:")
        for i in range(min(2, len(df_test_combined))):
            row = df_test_combined.iloc[i]
            print(f"\nTest Sample {i+1} ({row['data_source']}):")
            print(f"Question: {row['question'][:100]}...")
            print(f"Original answers: {row['answer']}")
            print(f"Golden answers: {row['golden_answers']}")

    else:
        # No test split - combine all data
        df_combined = pd.concat([df_factoid_processed, df_yesno_subset],
                                ignore_index=True)

        # Shuffle the combined dataset
        print("Shuffling combined dataset...")
        df_combined = df_combined.sample(frac=1,
                                         random_state=2).reset_index(drop=True)

        # Verify final composition
        factoid_count = (df_combined['data_source'] == 'bioasq_factoid').sum()
        yesno_count = (df_combined['data_source'] == 'bioasq_yesno').sum()
        final_composition = yesno_count / len(df_combined)

        print(f"Combined dataset: {len(df_combined)} samples")
        print(
            f"  - Factoid: {factoid_count} samples ({factoid_count/len(df_combined):.1%})"
        )
        print(f"  - Yesno: {yesno_count} samples ({final_composition:.1%})")

        # Create dataset name
        pcnt_int = int(pcnt * 100)
        dataset_name = f"bioasq_trainv0_factoidyesno_pcnt{pcnt_int}_n{n_samples}"

        # Upload to HuggingFace
        save_or_upload_dataset(df_combined, dataset_name)

        print("✅ Dataset processing complete!")
        print(f"Dataset uploaded as: {dataset_name}")

        # Print sample of results
        print("\nSample results:")
        for i in range(min(3, len(df_combined))):
            row = df_combined.iloc[i]
            print(f"\nSample {i+1} ({row['data_source']}):")
            print(f"Question: {row['question'][:100]}...")
            print(f"Original answers: {row['answer']}")
            print(f"Golden answers: {row['golden_answers']}")


def main():
    """Main function to process the dataset."""
    parser = argparse.ArgumentParser(
        description="Build training dataset v0 for BioASQ")
    parser.add_argument("--n_samples",
                        type=int,
                        required=True,
                        default=100,
                        help="Number of samples to process")
    parser.add_argument("--batch_size",
                        type=int,
                        default=50,
                        help="Batch size for API calls (default: 50)")
    parser.add_argument(
        "--n_test",
        type=int,
        default=None,
        help="Number of samples to use as test set (default: None)")
    parser.add_argument(
        "--is_yesno",
        action="store_true",
        help="Filter for yes/no questions instead of factoid questions")
    parser.add_argument("--factoid_yesno",
                        action="store_true",
                        help="Build combined factoid+yesno dataset")
    parser.add_argument(
        "--pcnt",
        type=float,
        default=0.2,
        help=
        "Percentage of factoid training samples to use for yesno training size (default: 0.2)"
    )

    args = parser.parse_args()

    if args.factoid_yesno:
        # Call the new combined dataset building function
        build_bioasq_factoid_yesno_dataset(n_samples=args.n_samples,
                                           n_test=args.n_test,
                                           batch_size=args.batch_size,
                                           pcnt=args.pcnt)
    else:
        # Call the original dataset building function
        build_bioasq_dataset(n_samples=args.n_samples,
                             n_test=args.n_test,
                             batch_size=args.batch_size,
                             is_yesno=args.is_yesno)


if __name__ == "__main__":
    main()
