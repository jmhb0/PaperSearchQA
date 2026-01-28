#!/usr/bin/env python3
"""
Create comprehensive BioASQ dataset with all question types organized by split.

This script creates a HuggingFace dataset where each BioASQ question type
(factoid, yesno, summary, list) is represented as a separate split.

IMPORTANT - DATA SOURCE AND LICENSE:
    This script uses the BioASQ-taskb dataset. BioASQ data is available for
    research purposes under specific terms:

    - Register at: http://bioasq.org/
    - License: BioASQ datasets are available for research and educational purposes
    - Attribution: Must cite BioASQ papers and acknowledge the dataset
    - Commercial use: Requires permission from BioASQ organizers

    BioASQ Citation:
        Tsatsaronis, G., Balikas, G., Malakasiotis, P., Partalas, I., Zschunke, M.,
        Alvers, M.R., Weissenborn, D., Krithara, A., Petridis, S., Polychronopoulos, D.
        and Almirantis, Y., 2015. An overview of the BIOASQ large-scale biomedical
        semantic indexing and question answering competition. BMC bioinformatics, 16(1), pp.1-28.

Usage:
    # Save locally only (default)
    python create_bioasq_all_types.py

    # Save locally and upload to HuggingFace
    export ENABLE_HF_UPLOAD=1
    export HF_USERNAME=your_username
    export HF_TOKEN=your_token
    python create_bioasq_all_types.py
"""

import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from typing import Dict

# Configuration
ENABLE_HF_UPLOAD = os.environ.get("ENABLE_HF_UPLOAD", "0") == "1"
HF_USERNAME = os.environ.get("HF_USERNAME", "jmhb")
OUTPUT_DIR = "output"
DATASET_NAME = "BioASQ"

# BioASQ question types (order matters for consistency)
QUESTION_TYPES = ['factoid', 'yesno', 'summary', 'list']


def create_bioasq_all_types_dataset() -> DatasetDict:
    """
    Create BioASQ dataset with all question types as separate splits.

    Returns:
        DatasetDict with splits for each question type
    """
    print("="*80)
    print("Creating BioASQ All Question Types Dataset")
    print("="*80)

    # Load the source dataset
    print("\nLoading BioASQ-taskb dataset...")
    source_dataset = load_dataset("jmhb/BioASQ-taskb")
    df = source_dataset['train'].to_pandas()

    # Drop index column if it exists
    if '__index_level_0__' in df.columns:
        df = df.drop(columns=['__index_level_0__'])

    print(f"Loaded {len(df)} total samples")
    print(f"\nQuestion type distribution:")
    print(df['type'].value_counts().sort_index())

    # Create splits dictionary
    splits_dict = {}

    for question_type in QUESTION_TYPES:
        print(f"\nProcessing '{question_type}' questions...")

        # Filter for this question type
        df_filtered = df[df['type'] == question_type].copy()
        print(f"  Found {len(df_filtered)} samples")

        # Reset index
        df_filtered = df_filtered.reset_index(drop=True)

        # Convert to Dataset
        dataset = Dataset.from_pandas(df_filtered, preserve_index=False)
        splits_dict[question_type] = dataset

        print(f"  Columns: {list(dataset.column_names)}")

    # Create DatasetDict
    dataset_dict = DatasetDict(splits_dict)

    print("\n" + "="*80)
    print("Dataset Summary")
    print("="*80)
    print(f"Total splits: {len(dataset_dict)}")
    for split_name, split_data in dataset_dict.items():
        print(f"  - {split_name}: {len(split_data)} samples")

    return dataset_dict


def save_and_upload_dataset(dataset: DatasetDict):
    """
    Save dataset locally and optionally upload to HuggingFace.

    Args:
        dataset: DatasetDict to save/upload
    """
    dataset_name_with_prefix = f"{HF_USERNAME}/{DATASET_NAME}"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save each split as parquet locally
    print("\n" + "="*80)
    print("Saving Dataset Locally")
    print("="*80)

    for split_name, split_data in dataset.items():
        output_file = os.path.join(
            OUTPUT_DIR,
            f"{dataset_name_with_prefix.replace('/', '_')}_{split_name}.parquet"
        )
        split_data.to_parquet(output_file)
        print(f"✅ Saved '{split_name}' split ({len(split_data)} samples) to: {output_file}")

    # Upload to HuggingFace if enabled
    if ENABLE_HF_UPLOAD:
        print("\n" + "="*80)
        print("Uploading to HuggingFace")
        print("="*80)
        print(f"Dataset: {dataset_name_with_prefix}")

        try:
            dataset.push_to_hub(
                dataset_name_with_prefix,
                private=False,
                token=os.getenv("HF_TOKEN")
            )
            print(f"✅ Uploaded to: https://huggingface.co/datasets/{dataset_name_with_prefix}")
            print(f"\nℹ️  Remember to:")
            print(f"   1. Add README.md with proper BioASQ attribution")
            print(f"   2. Add dataset card with license information")
            print(f"   3. Cite BioASQ papers in the documentation")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    else:
        print("\n" + "="*80)
        print("Upload Disabled")
        print("="*80)
        print("ℹ️  To upload to HuggingFace, set:")
        print("   export ENABLE_HF_UPLOAD=1")
        print("   export HF_USERNAME=your_username")
        print("   export HF_TOKEN=your_token")


def print_sample_data(dataset: DatasetDict):
    """
    Print sample data from each split.

    Args:
        dataset: DatasetDict to sample from
    """
    print("\n" + "="*80)
    print("Sample Data")
    print("="*80)

    for split_name, split_data in dataset.items():
        print(f"\n--- {split_name.upper()} (sample 1/{len(split_data)}) ---")
        sample = split_data[0]
        print(f"Question: {sample['question'][:150]}...")
        print(f"Type: {sample['type']}")

        # Print answer based on type
        answer = sample.get('answer', sample.get('exact_answer', 'N/A'))
        if isinstance(answer, list):
            if len(answer) > 0:
                print(f"Answer: {answer[0] if len(answer) == 1 else answer[:2]}...")
        else:
            print(f"Answer: {str(answer)[:100]}...")

        # Print ideal answer if available
        if 'ideal_answer' in sample and sample['ideal_answer']:
            ideal = sample['ideal_answer']
            if isinstance(ideal, list) and len(ideal) > 0:
                print(f"Ideal answer: {ideal[0][:100]}...")
            elif isinstance(ideal, str):
                print(f"Ideal answer: {ideal[:100]}...")


def main():
    """Main execution function."""
    print("\nBioASQ All Question Types Dataset Creator")
    print("This script creates a dataset with all BioASQ question types")
    print("organized as separate splits (factoid, yesno, summary, list)")

    # Create dataset
    dataset = create_bioasq_all_types_dataset()

    # Print samples
    print_sample_data(dataset)

    # Save and optionally upload
    save_and_upload_dataset(dataset)

    print("\n" + "="*80)
    print("✅ Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the saved parquet files in the 'output/' directory")
    if ENABLE_HF_UPLOAD:
        print(f"2. Add README.md to: https://huggingface.co/datasets/{HF_USERNAME}/{DATASET_NAME}")
        print("3. Ensure proper BioASQ attribution in dataset card")
        print("4. Add license information")
    else:
        print("2. Set ENABLE_HF_UPLOAD=1 to upload to HuggingFace")
    print("5. Cite BioASQ papers in any publications using this data")


if __name__ == "__main__":
    main()
