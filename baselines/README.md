# Baselines

This directory contains baseline implementations for evaluating question answering performance on the PaperSearchQA dataset.

## Available Baselines

### RAG (Retrieval-Augmented Generation)
**Location**: `baselines/rag/`

Standard RAG and SearchR1 baseline implementations that retrieve relevant documents and generate answers using the retrieved context.

**Key Files:**
- `batch_rag.py` - Batch RAG implementation with vLLM
- `run_inference.py` - Main inference script supporting multiple methods (direct, CoT, RAG)
- `infer.py` - SearchR1 inference utilities
- `infer_searchr1.py` - BatchSearchR1 for multi-turn search
- `qa_em.py` - Exact match evaluation module

**Features:**
- Multiple inference methods: Direct, Chain-of-Thought, RAG
- Batch retrieval and generation for efficiency
- Caching system for LLM responses
- Support for multiple retrieval backends (BM25, dense retrieval)
- Exact match and LLM judge evaluation
- SearchR1 multi-turn search-augmented inference

**Usage:**
```bash
# Run RAG baseline
python baselines/rag/run_inference.py \
    --method rag \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA \
    --rag_top_k 3 \
    --retriever_type bm25

# Run Direct inference (no retrieval)
python baselines/rag/run_inference.py \
    --method direct \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA

# Run Chain-of-Thought
python baselines/rag/run_inference.py \
    --method cot \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA
```

**Requirements:**
- Running retrieval server on localhost:8001 (see search-r1/ for retrieval server)
- vLLM for fast batch inference

## Prerequisites

### Retrieval Server
All baselines require a running retrieval server. See the `search-r1/` directory for setup:

```bash
# Start retrieval server (from search-r1/)
bash retrieval_launch.sh  # or retrieval_launch_bm25.sh for BM25
```

The server should be accessible at `http://localhost:8001/retrieve`

### Corpus Data
Baselines use the PubMed abstract corpus. See `data_generation/` for corpus creation:

```bash
# Create corpus from allMeSH data
python data_generation/core_pipeline/make_pubmed_corpus.py
```

## Dependencies

Install required packages:

```bash
# Core dependencies
pip install torch transformers datasets pandas numpy tqdm vllm

# For evaluation
pip install lmdb filelock

# For retrieval (verl package - install from Search-R1 project)
# See search-r1/ directory for verl installation
```

## Evaluation Datasets

Baselines are evaluated on:

1. **PaperSearchQA** - Main dataset (20K train, 5K test)
   - HuggingFace: `PaperSearchQA/PaperSearchQA`

2. **BioASQ Factoid** - Held-out test set
   - HuggingFace: `PaperSearchQA/BioASQ_factoid`

## Running Evaluations

### Direct Inference (No Retrieval)
```bash
python baselines/rag/run_inference.py \
    --method direct \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA \
    --first_n 100
```

### Chain-of-Thought
```bash
python baselines/rag/run_inference.py \
    --method cot \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA
```

### RAG
```bash
python baselines/rag/run_inference.py \
    --method rag \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA \
    --rag_top_k 3 \
    --retriever_type bm25 \
    --corpus_filename pubmed.jsonl
```

## Output

Results are saved in `results/eval/run_inference/` with:
- CSV file with predictions and scores
- TXT file with summary statistics
- Cached responses in `cache/infer_cache.lmdb`

## Performance Notes

- **Caching**: First run will be slow, subsequent runs use cache
- **Batch Size**: Adjust based on GPU memory
- **Retrieval**: BM25 is faster but less accurate than dense retrieval
- **Search-O1**: More searches improves accuracy but increases cost

## Citation

If you use these baselines, please cite:

```bibtex
@inproceedings{papersearchqa2025,
  title={PaperSearchQA: A Dataset for Question Answering over Scientific Literature},
  author={Your Name},
  booktitle={EACL},
  year={2025}
}
```

## Notes

- RAG and Search-O1 baselines adapt methodologies from PaperQA for biomedical QA
- Answer extraction uses `<answer></answer>` tags for consistency
- Exact match evaluation uses the same methodology as the PaperSearchQA paper
