# PaperQA Baseline

This directory contains a modified implementation of [PaperQA](https://github.com/Future-House/paper-qa) adapted for the PaperSearchQA evaluation pipeline.

**Note**: Significant changes were required to integrate PaperQA with our retrieval infrastructure and evaluation framework. The implementation differs from the original PaperQA in several ways:

- Custom retrieval integration with our BM25/dense retrieval servers
- Modified embedding handling (HTTP-based embeddings with OpenAI LLM)
- Ollama support for local model inference
- Batch evaluation scripts for large-scale testing
- Adapted scoring to match our Exact Match (EM) evaluation metrics

## Setup

**Prerequisites:**
1. Start retrieval server (see `search-r1/` directory):
   ```bash
   cd search-r1 && bash retrieval_launch_pubmed_bm25.sh
   ```

2. Start vLLM server:
   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8003
   ```

3. Set environment variables:
   ```bash
   export RETRIEVE_URL="http://localhost:8001/retrieve"
   export RETRIEVE_BEARER_TOKEN="..."  # optional
   ```

**Install dependencies:**
```bash
pip install paper-qa==4.4.0 langchain-core>=0.2.4 httpx openai>=1.0.0 diskcache
```

## Usage

**Batch evaluation** (recommended for benchmarking):
```bash
python baselines/paperqa/run_batch_evaluation.py
```

Configuration options (edit at top of script):
- `DATASET_NAME` - Dataset to evaluate (default: `"PaperSearchQA/BioASQ_factoid"`)
- `N_SAMPLES` - Number of questions to process (default: 250)
- `MAX_CONCURRENT_WORKERS` - Parallel workers (default: 50)
- `MAX_SOURCES` - Retrieved document chunks per query (default: 3)

**Single query** (for testing):
```bash
python baselines/paperqa/run_retrieval.py
```

## Files

- `run_batch_evaluation.py` - Batch evaluation with caching and parallel processing
- `run_retrieval.py` - Basic retrieval-based inference (single query)
- `run_retrieval_ollama.py` - Ollama-based inference variant
- `run_paperqa_http_embedding_but_openai_llm.py` - Hybrid embedding/LLM setup
- `compute_accuracy.py` - Accuracy computation using EM scoring

## Citation

If you use PaperQA, please cite the original work:
```bibtex
@software{paperqa,
  title = {PaperQA},
  author = {Future House},
  year = {2024},
  url = {https://github.com/Future-House/paper-qa}
}
```
