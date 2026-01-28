"""
Exact Match (EM) scoring for QA evaluation.

This module provides exact match evaluation for question answering,
adapted from the verl.utils.reward_score.qa_em module used in Search-R1.
"""

import re
from typing import Dict, List, Union


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    # Convert to lowercase
    text = text.lower().strip()

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def extract_answer_from_text(text: str) -> str:
    """Extract answer from text, looking for <answer> tags."""
    if not text:
        return ""

    # Find all <answer>...</answer> tags
    pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)

    if matches:
        # Return the last answer (in case there are multiple)
        return matches[-1].strip()

    # If no tags found, return the original text
    return text.strip()


def compute_score_em(
    solution_str: str,
    ground_truth: Dict[str, List[str]],
    method: str = 'strict',
    format_score: float = 0.0,
    score: float = 1.0
) -> float:
    """
    Compute exact match score between solution and ground truth answers.

    Args:
        solution_str: The predicted answer, possibly with <answer> tags
        ground_truth: Dict with 'target' key containing list of acceptable answers
        method: Scoring method ('strict' for exact match)
        format_score: Score to return if answer format is invalid
        score: Score to return for a match

    Returns:
        float: score if match found, format_score if no match
    """
    # Extract answer from solution string
    predicted = extract_answer_from_text(solution_str)

    if not predicted:
        return format_score

    # Normalize predicted answer
    normalized_pred = normalize_answer(predicted)

    # Get target answers
    target_answers = ground_truth.get('target', [])

    if not target_answers:
        return format_score

    # Check if prediction matches any target answer
    for target in target_answers:
        normalized_target = normalize_answer(str(target))

        if normalized_pred == normalized_target:
            return score

    # No match found
    return format_score


# For backwards compatibility
def compute_score_em_simple(prediction: str, ground_truth_list: List[str]) -> float:
    """
    Simplified EM scoring with direct prediction and list of answers.

    Args:
        prediction: The predicted answer (plain text)
        ground_truth_list: List of acceptable answers

    Returns:
        float: 1.0 if match, 0.0 otherwise
    """
    normalized_pred = normalize_answer(prediction)

    for target in ground_truth_list:
        normalized_target = normalize_answer(str(target))
        if normalized_pred == normalized_target:
            return 1.0

    return 0.0


if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "pred": "<answer>dystrophin</answer>",
            "target": ["dystrophin", "DMD protein"],
            "expected": 1.0
        },
        {
            "pred": "<answer>The dystrophin protein</answer>",
            "target": ["dystrophin"],
            "expected": 1.0
        },
        {
            "pred": "<answer>CFTR</answer>",
            "target": ["cystic fibrosis transmembrane conductance regulator", "CFTR"],
            "expected": 1.0
        },
        {
            "pred": "<answer>wrong answer</answer>",
            "target": ["correct answer"],
            "expected": 0.0
        },
    ]

    print("Running QA-EM tests...")
    for i, case in enumerate(test_cases):
        ground_truth = {"target": case["target"]}
        score = compute_score_em(case["pred"], ground_truth)
        status = "✓" if score == case["expected"] else "✗"
        print(f"{status} Test {i+1}: pred='{extract_answer_from_text(case['pred'])}', "
              f"targets={case['target']}, score={score} (expected {case['expected']})")
