#!/usr/bin/env python3
"""
Script to compute per-category accuracy from evaluation results.

This script processes CSV files in a results directory and computes
accuracy metrics per category based on the 'cat' column.
"""

import argparse
import pandas as pd
import os
import glob
from pathlib import Path


def compute_per_category_accuracy(csv_file):
    """Compute per-category accuracy from a single CSV file."""
    try:
        df = pd.read_csv(csv_file)

        # Check if required columns exist
        if 'cat' not in df.columns or 'em_score' not in df.columns:
            print(f"Warning: Required columns 'cat' or 'em_score' not found in {csv_file}")
            return None

        # Compute per-category accuracy
        category_stats = df.groupby('cat')['em_score'].agg([
            'count',  # total questions
            'sum',    # correct answers
            'mean'    # accuracy
        ]).round(4)

        category_stats.columns = ['total_questions', 'correct_answers', 'accuracy']
        category_stats = category_stats.sort_values('accuracy', ascending=False)

        # Add overall accuracy
        overall_accuracy = df['em_score'].mean()

        return category_stats, overall_accuracy

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compute per-category accuracy from evaluation results')
    parser.add_argument(
        '--src_path',
        type=str,
        default="results/eval/run_inference/jmhb-papersearchr1/",
        help='Path to the results folder containing CSV files'
    )

    args = parser.parse_args()

    src_path = Path(args.src_path)
    if not src_path.exists():
        print(f"Error: Source path {src_path} does not exist")
        return

    # Use the same directory as the source for output
    output_dir = src_path

    print(f"Processing CSV files in: {src_path}")
    print(f"Output directory: {output_dir}")

    # Find all CSV files in the source directory
    csv_files = list(src_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {src_path}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # Process each CSV file
    all_results = {}

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        result = compute_per_category_accuracy(csv_file)
        if result is None:
            continue

        category_stats, overall_accuracy = result

        # Create output filename
        output_filename = csv_file.stem + "_per_category.csv"
        output_path = output_dir / output_filename

        # Save per-category results
        category_stats.to_csv(output_path)

        # Store results for summary
        all_results[csv_file.name] = {
            'category_stats': category_stats,
            'overall_accuracy': overall_accuracy
        }

        print(f"  Overall accuracy: {overall_accuracy:.4f}")
        print(f"  Saved to: {output_path}")
        print(f"  Top categories by accuracy:")
        for cat, row in category_stats.head(3).iterrows():
            print(f"    {cat}: {row['accuracy']:.4f} ({int(row['correct_answers'])}/{int(row['total_questions'])})")

    # Create summary files (both TXT and CSV)
    summary_txt_path = output_dir / "summary_per_category.txt"
    summary_csv_path = output_dir / "summary_per_category.csv"

    # Create text summary
    with open(summary_txt_path, 'w') as f:
        f.write("Per-Category Accuracy Summary\n")
        f.write("=" * 50 + "\n\n")

        for csv_name, results in all_results.items():
            f.write(f"File: {csv_name}\n")
            f.write(f"Overall Accuracy: {results['overall_accuracy']:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write("Category-wise Results:\n")

            for cat, row in results['category_stats'].iterrows():
                f.write(f"  {cat:<35} {row['accuracy']:.4f} ({int(row['correct_answers'])}/{int(row['total_questions'])})\n")
            f.write("\n")

    # Create CSV summary
    summary_data = []
    for csv_name, results in all_results.items():
        # Add overall accuracy row
        summary_data.append({
            'file': csv_name,
            'category': 'OVERALL',
            'accuracy': results['overall_accuracy'],
            'correct_answers': int(results['category_stats']['correct_answers'].sum()),
            'total_questions': int(results['category_stats']['total_questions'].sum())
        })

        # Add per-category rows
        for cat, row in results['category_stats'].iterrows():
            summary_data.append({
                'file': csv_name,
                'category': cat,
                'accuracy': row['accuracy'],
                'correct_answers': int(row['correct_answers']),
                'total_questions': int(row['total_questions'])
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv_path, index=False)

    # Create unstacked/pivoted CSV with categories as columns
    pivot_csv_path = output_dir / "summary_per_category_pivot.csv"

    # Filter out OVERALL rows for the pivot table
    category_data = summary_df[summary_df['category'] != 'OVERALL'].copy()

    # Create pivot table with files as rows and categories as columns
    pivot_df = category_data.pivot(index='file', columns='category', values='accuracy')

    # Add OVERALL column back
    overall_data = summary_df[summary_df['category'] == 'OVERALL'].set_index('file')['accuracy']
    pivot_df['OVERALL'] = overall_data

    # Reorder columns to put OVERALL first
    cols = ['OVERALL'] + [col for col in pivot_df.columns if col != 'OVERALL']
    pivot_df = pivot_df[cols]

    # Save the pivoted data
    pivot_df.to_csv(pivot_csv_path)

    print(f"\nSummary saved to: {summary_txt_path}")
    print(f"Summary CSV saved to: {summary_csv_path}")
    print(f"Summary pivot CSV saved to: {pivot_csv_path}")
    print(f"Individual per-category results saved to: {output_dir}")


if __name__ == "__main__":
    main()