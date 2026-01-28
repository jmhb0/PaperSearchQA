# Data Generation Pipeline

This folder contains the data construction pipeline for PaperSearchQA - the primary artifact of this project.

## Overview

The pipeline generates question-answer pairs from PubMed abstracts for biomedical question answering with retrieval. The main dataset is generated from scratch by prompting GPT-4.1 with PubMed abstracts. The BioASQ dataset is processed separately as a test set.

**Final datasets:**
1. **PaperSearchQA/PaperSearchQA** - Main dataset (generated from PubMed abstracts)
2. **PaperSearchQA/BioASQ_factoid** - BioASQ test set with golden answers

## Quick Start

**Before running the full pipeline, validate it works with the smoke test:**

```bash
python minimal_test.py
```

This quick test (5 abstracts, ~15 Q&A pairs, ~$0.50, 3-5 minutes) verifies the entire pipeline end-to-end before running expensive production-scale generation. See [SMOKE_TEST.md](SMOKE_TEST.md) for details.

## Core Pipeline Files

### Main Generation Pipeline

1. **`allMesh_to_parquet.py`** - Convert allMeSH JSON to indexed parquet format
2. **`make_pubmed_corpus.py`** - Create PubMed JSONL corpus from allMeSH data
3. **`generate_questions_from_abstracts.py`** - **MAIN SCRIPT** - Generate Q&A pairs from abstracts
4. **`api.py`** - LLM API wrapper with LMDB caching

### BioASQ Processing

5. **`process_bioasq_golden_answers.py`** - Add golden_answers to BioASQ dataset (see [BioASQ Data Source](#bioasq-data-source) below)
6. **`create_bioasq_factoid.py`** - Create BioASQ factoid test set

### Auxiliary Files

- `bioasq_inference.py` - BioASQ inference utilities
- `make_bioasq_dataset.py` - Earlier BioASQ processing (superseded)
- `inference_qwen_local.py` - Local Qwen model inference utilities
- `test_retrieval.py` - Test retrieval functionality

## Main Pipeline: generate_questions_from_abstracts.py

This is the **primary script** that generates the PaperSearchQA dataset.

**Process:**
1. Loads PubMed abstracts from `allMeSH_2022.parquet`
2. Samples n_samples abstracts randomly (seed=42)
3. Generates Q&A pairs using GPT-4.1 with category-aware prompts
4. Filters out answers containing "this study" (too abstract-specific)
5. Generates `golden_answers` (synonym lists) using GPT-4.1
6. Optionally applies paraphrasing (GPT-4o-mini) to questions
7. Creates train/test split
8. Saves to `output/` directory as Parquet files
9. Optionally uploads to HuggingFace if `HF_USERNAME` is set

**LLM Models:**
- `openai/gpt-4.1` - Question generation and synonym generation
- `openai/gpt-4o-mini` - Question paraphrasing (optional)

**Question Categories (9 types):**
1. Genetic inheritance & disease-linked mutations
2. Therapeutics, indications & clinical evidence
3. Protein function, localization & signaling
4. Experimental & computational methods
5. Disease causation & pathogens
6. Biomarkers & diagnostic tests
7. Bioinformatics databases & resources
8. Clinical grading & diagnostic scales
9. Anatomical/cellular structures & localization

**Output:**
- Intermediate: `output/<dataset_name>_initial.csv` (before golden_answers)
- Final: `output/<dataset_name>.csv` (with golden_answers + paraphrasing)
- Parquet: `output/<dataset_name>_{train,test}.parquet`
- Optional: Pushes to `<HF_USERNAME>/<dataset_name>` on HuggingFace

**Dataset naming:**
- Format: `PaperSearchRL_v{template}_gv{golden_template}_n{samples}_test{test_size}`
- With paraphrasing: `..._parav{para_template}pcnt{percentage}`
- Example: `PaperSearchRL_v4_gv2_n20000_test5000_parav1pcnt50`

## Data Flow Diagram

```
allMeSH_2022.json (27GB, from BioASQ)
    ↓
[allMesh_to_parquet.py]
    ↓
allMeSH_2022.parquet (13GB) + indices (1.2GB)
    ↓
[make_pubmed_corpus.py]
    ↓
pubmed.jsonl (23GB) + lookup (356MB)
    ↓
[generate_questions_from_abstracts.py]
    ├─→ Sample abstracts
    ├─→ Generate Q&A (GPT-4.1)
    ├─→ Filter "this study" answers
    ├─→ Generate golden_answers (GPT-4.1)
    ├─→ Paraphrase questions (GPT-4o-mini, optional)
    ├─→ Train/test split
    └─→ Save to output/*.parquet + optional HF upload

Separately:
BioASQ-taskb (requires registration)
    ↓
[process_bioasq_golden_answers.py]
    └─→ Add golden_answers (GPT-4o-mini)
        └─→ output/bioasq_*.parquet + optional HF upload
```

## Source Data

### PubMed Abstracts (allMeSH_2022.json)

**Source:** BioASQ 2022 challenge
**Paper:** https://ceur-ws.org/Vol-3180/paper-10.pdf
**Size:** 27GB JSON
**License:** BioASQ license with attribution

**Attribution:**
```
The PubMed abstract corpus (allMeSH_2022.json) was obtained from the BioASQ 2022 challenge.
Please cite: BioASQ: A Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering
```

**Contents:**
- ~36M PubMed abstracts with MeSH annotations
- Fields: pmid, title, abstractText, year, meshMajor

**Download:** Contact BioASQ organizers or use existing processed versions on HuggingFace

### Processed Versions on HuggingFace

We recommend using the pre-processed parquet version (saves time):

**Option 1: Download allMeSH parquet (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("jmhb/allMeSH_2022_indexed")  # TODO: Upload this
# Contains: allMeSH_2022.parquet + indices
```

**Option 2: Download JSONL corpus:**
```python
dataset = load_dataset("jmhb/pubmed_abstract_corpus")  # Already on HF
# Contains: 48 Arrow files (23GB total)
```

## BioASQ Data Source

The BioASQ dataset is **NOT publicly available** and requires registration:

1. Register at http://bioasq.org/
2. Download BioASQ Task B data
3. To run `process_bioasq_golden_answers.py`, you must:
   - Upload BioASQ data to a private HuggingFace dataset
   - Update `load_dataset("jmhb/BioASQ-taskb")` calls in the script

**For users:** The processed BioASQ test set (with golden_answers added) is available at:
- https://huggingface.co/datasets/PaperSearchQA/BioASQ_factoid

The `process_bioasq_golden_answers.py` script is provided for documentation purposes to show how golden_answers were generated.

## Installation

```bash
pip install -r requirements.txt
```

**Required packages:**
- `datasets` - HuggingFace datasets library
- `pandas`, `numpy` - Data manipulation
- `lmdb` - Fast key-value cache
- `filelock` - Thread-safe file locking
- `openai` - OpenAI API client (for OpenRouter)
- `httpx` - HTTP client
- `pyarrow` - Parquet support
- `tqdm` - Progress bars

## Configuration

### HuggingFace Uploads (Optional)

By default, all scripts **save datasets locally** to the `output/` directory as Parquet files.

To enable uploads to HuggingFace Hub, set the `HF_USERNAME` environment variable:

```bash
export HF_USERNAME="your_username"
export HF_TOKEN="your_hf_token"  # For authentication

# Now scripts will save locally AND upload to HuggingFace
python generate_questions_from_abstracts.py
```

**Output behavior:**
- **Always**: Saves to `output/<dataset_name>_{train,test}.parquet`
- **If HF_USERNAME set**: Also uploads to `<HF_USERNAME>/<dataset_name>` on HuggingFace Hub
- **If HF_USERNAME not set**: Only saves locally with a reminder message

This design ensures the code works out-of-the-box without requiring HuggingFace credentials.

### API Keys

Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your_key_here"
```

## Cache Files

All caches are LMDB databases stored in `cache/`:

| Cache | Size | Purpose |
|-------|------|---------|
| `llm_cache.lmdb` | 205MB | Main LLM API response cache |

The cache contains API responses that cost money to generate. It enables:
- **Cost savings**: Avoid redundant API calls (~$100+ saved)
- **Speed**: Instant retrieval for cached prompts
- **Reproducibility**: Same inputs always return same outputs

### Extracting the Cache

The repository includes `cache.tar.gz` with pre-cached LLM responses. Extract it before running data generation:

```bash
tar -xzvf cache.tar.gz
```

This will create the `cache/` directory with the LMDB cache files, significantly reducing API costs when running the pipeline.

## Usage Examples

### Generate PaperSearchQA Dataset

```bash
# Edit generate_questions_from_abstracts.py to set parameters at bottom:
# n_samples = 20000
# n_test = 5000
# key = 5  # Template version
# golden_key = 3  # Golden answers template version
# do_paraphrase = True
# paraphrase_pcnt = 0.5

python generate_questions_from_abstracts.py

# Output: output/PaperSearchRL_v5_gv3_n20000_test5000_parav1pcnt50_{train,test}.parquet
```

### Process BioASQ (requires private data source)

```bash
python process_bioasq_golden_answers.py --n_samples 1609 --n_test 100
```

## Costs and Time Estimates

For a dataset with 20,000 samples:
- **Question generation**: ~$100-150 (GPT-4.1)
- **Golden answers**: ~$30-50 (GPT-4.1)
- **Paraphrasing** (50%): ~$20-30 (GPT-4o-mini)
- **Total cost**: ~$150-230
- **Time**: ~4-6 hours with caching

Caching reduces costs dramatically on reruns.

## LLM Models Used

| Script | Model | Purpose |
|--------|-------|---------|
| `generate_questions_from_abstracts.py` | `openai/gpt-4.1` | Question generation |
| `generate_questions_from_abstracts.py` | `openai/gpt-4.1` | Golden answers (synonyms) |
| `generate_questions_from_abstracts.py` | `openai/gpt-4o-mini` | Question paraphrasing |
| `process_bioasq_golden_answers.py` | `openai/gpt-4o-mini` | Synonym generation |

All models accessed via OpenRouter API.

## Notes

- **Caches are valuable**: The LLM cache contains ~205MB of API responses that cost money to generate. Include it in releases when possible.
- **Random seeds**: Scripts use fixed random seeds (42) for reproducibility.
- **Batching**: API calls are batched (typically 10 concurrent) to balance speed and rate limits.
- **Error handling**: Scripts include robust error handling for API failures and parsing errors.
- **Local-first design**: All scripts save outputs locally by default; HuggingFace uploads are optional via `HF_USERNAME` environment variable.
- **Paraphrasing**: Increases dataset size and diversity but adds cost (~50% more questions if `paraphrase_pcnt=0.5`)

## Citation

If you use this data generation pipeline, please cite:

```bibtex
@inproceedings{papersearchqa2025,
  title={PaperSearchQA: Training Language Models to Search and Answer Questions over Scientific Papers},
  author={Burgess, James and Hansen, Jan Niklas and Peng, Duo and Zhang, Yuhui and Lozano Garcia, Eduardo Alejandro and Sun, Min and Lundberg, Emma and Yeung, Serena},
  booktitle={EACL},
  year={2025}
}
```

And acknowledge the BioASQ source data:
```bibtex
@article{bioasq,
  title={BioASQ: A Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering},
  author={Tsatsaronis, George and others},
  journal={BMC Bioinformatics},
  year={2015}
}
```
