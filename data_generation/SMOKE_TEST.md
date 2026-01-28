# Smoke Test for Data Generation Pipeline

Quick validation test for the PaperSearchQA data generation pipeline.

## What It Does

The smoke test generates a small dataset (5 abstracts, ~15 Q&A pairs) to verify the entire pipeline works correctly before running expensive production-scale generation.

**Pipeline steps tested:**
1. Downloads PubMed corpus from HuggingFace (`jmhb/pubmed_bioasq_2022`)
2. Generates questions using GPT-4 (3 questions per abstract)
3. Filters out "this study" references
4. Generates golden answers (synonym lists)
5. Creates train/test split
6. Saves as Parquet files

## Quick Start

**Prerequisites:**
```bash
# Required: OpenRouter API key
export OPENROUTER_API_KEY="your_key_here"

# Install dependencies
cd data_generation
pip install -r requirements.txt
```

**Run the test:**
```bash
python minimal_test.py
```

**Expected:**
- Time: 3-5 minutes
- Cost: ~$0.50
- Output: ~15 Q&A pairs in `output/` directory

## Memory Requirements

The test loads a 13GB parquet file into memory. You need:
- **Minimum:** 16GB RAM
- **Recommended:** Run in a Slurm session with 32GB+ allocation

Check your limit:
```bash
ulimit -m  # Should show > 16000000 (16GB in KB)
```

If running in a restricted Slurm job, request more memory:
```bash
srun --mem=32G --pty bash
```

## Output

The test creates:
```
output/
  your_username_minimal-test_train.parquet  (~8 KB, 13 samples)
  your_username_minimal-test_test.parquet   (~5 KB, 2 samples)

results/generate_dataset_from_abstracts/
  minimal-test_initial.csv  (before golden answers)
  minimal-test.csv          (final with synonyms)
```

## Dataset Structure

```python
{
    'question': str,              # The question
    'answer': str,                # The answer
    'golden_answers': List[str],  # Answer + synonyms
    'cat_num': str,               # Category number (1-10)
    'cat': str,                   # Category description
    'pmid': str,                  # PubMed ID
    'paper_title': str,           # Paper title
}
```

## Troubleshooting

**Process crashes during parquet loading:**
- Check memory: `free -h` and `ulimit -m`
- Solution: Run in Slurm session with more memory

**"ModuleNotFoundError: No module named 'datasets'":**
- Solution: `pip install -r requirements.txt`

**Download fails with 404 error:**
- The parquet file is at `data/allMeSH_2022.parquet` (not root)
- This is already handled in the script

## Next Steps

After successful smoke test:

1. **Review the output** in `output/` directory
2. **Run production generation** with more abstracts:
   ```bash
   cd core_pipeline
   # Edit generate_questions_from_abstracts.py:
   # n_samples = 20000
   # n_test = 5000
   python generate_questions_from_abstracts.py
   ```

Production run estimates:
- 20,000 abstracts → ~60,000 Q&A pairs
- Cost: ~$150-230
- Time: 4-6 hours

## Questions?

See the main documentation in `USAGE.md` for complete pipeline details.
