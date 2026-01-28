# Search-R1 Retrieval Infrastructure

This directory contains the retrieval infrastructure from [Search-R1](https://github.com/PeterGriffinJin/Search-R1), adapted for the PaperSearchQA project. It provides BM25 and dense retrieval capabilities over the PubMed corpus for baseline evaluation.

## Overview

Search-R1 is a reinforcement learning framework for training reasoning-and-searching interleaved LLMs. For PaperSearchQA, we use its retrieval components to:
- Index the PubMed corpus (16M abstracts)
- Launch BM25 and E5 dense retrieval servers
- Support RAG and SearchR1 baselines

**Full Search-R1 documentation**: https://github.com/PeterGriffinJin/Search-R1

## Quick Start

### 1. Installation

**Main search-r1 environment:**
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install flash-attn --no-build-isolation
```

**Retriever environment (for BM25/E5 servers):**
```bash
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

### 2. Launch Retrieval Server

**For BM25 retrieval (recommended for PaperSearchQA):**
```bash
conda activate retriever
bash retrieval_launch_pubmed_bm25.sh
```

**For E5 dense retrieval:**
```bash
conda activate retriever
bash retrieval_launch_pubmed_e5.sh
```

The server will launch on `http://0.0.0.0:8000` and listen for retrieval requests at the `/retrieve` endpoint.

**Important**: The retrieval server must be running before executing baselines or training scripts. All components (baselines, training scripts) are configured to connect to `http://127.0.0.1:8000/retrieve` by default.

### 3. Run Baselines

Once the retrieval server is running, you can run baselines from the parent directory:

```bash
# RAG baseline
python baselines/rag/run_inference.py \
    --method rag \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA

# SearchR1 baseline
python baselines/rag/run_inference.py \
    --method searchr1 \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA
```

## Directory Structure

```
search-r1/
├── search_r1/              # Core retrieval package
│   ├── llm_agent/         # LLM generation utilities
│   └── search/            # Retrieval servers and indexing
├── verl/                  # Reinforcement learning framework
├── scripts/               # Data processing utilities
├── retrieval_launch_*.sh  # Server launch scripts
├── install_*.sh          # Installation scripts
└── requirements.txt       # Python dependencies
```

## Key Components

### Retrieval Servers

- **`retrieval_launch_pubmed_bm25.sh`**: Launch BM25 retriever for PubMed corpus
- **`retrieval_launch_pubmed_e5.sh`**: Launch E5 dense retriever for PubMed corpus
- **`retrieval_launch.sh`**: Generic retrieval server launcher

### Search Package

- **`search_r1/search/retrieval_server.py`**: Main retrieval server implementation
- **`search_r1/search/index_builder.py`**: Corpus indexing utilities
- **`search_r1/search/build_index_pubmed.sh`**: Build BM25 index for PubMed

### Training Framework (Optional)

The `verl/` directory contains the full RL training framework from Search-R1. This is included for completeness but is not required for running PaperSearchQA baselines.

## Configuration

### Port Configuration

The retrieval server runs on **port 8000** by default. This is configured in:
- `search_r1/search/retrieval_server.py`: `PORT = 8000`
- All training scripts expect the server at `http://127.0.0.1:8000/retrieve`
- All baseline scripts expect the server at `http://127.0.0.1:8000/retrieve`

If you need to change the port, update:
1. `PORT` variable in `search_r1/search/retrieval_server.py`
2. `RETRIEVER_URL` in training scripts (if using RL training)
3. Retrieval URL in baseline scripts (if running baselines)

### Data Paths

The launch scripts (`retrieval_launch_pubmed_*.sh`) are pre-configured with paths:
- **Corpus**: `data/pubmed.jsonl`
- **BM25 index**: `data/pubmed_bm25/bm25`
- **E5 index**: `data/pubmed_e5/e5_Flat.index`

These paths point to the PubMed corpus and indices. See the PubMed Corpus section below for how to obtain the data.

## PubMed Corpus

The retrieval infrastructure is configured to work with the PubMed corpus:

- **HuggingFace**: `jmhb/pubmed_bioasq_2022`
- **Size**: 16 million abstracts
- **Format**: JSONL with fields: title, abstract, PMID, MeSH terms

The corpus is automatically used by the retrieval servers when launched with the `*_pubmed_*` scripts.

## Citation

If you use the Search-R1 retrieval infrastructure, please cite:

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

And for PaperSearchQA:

```bibtex
@misc{burgess2026papersearchqalearningsearchreason,
      title={PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR},
      author={James Burgess and Jan N. Hansen and Duo Peng and Yuhui Zhang and Alejandro Lozano and Min Woo Sun and Emma Lundberg and Serena Yeung-Levy},
      year={2026},
      eprint={2601.18207},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.18207},
}
```

## Acknowledgments

This retrieval infrastructure is adapted from [Search-R1](https://github.com/PeterGriffinJin/Search-R1). We thank the Search-R1 team for their open-source contribution.

## License

Apache 2.0 License (inherited from Search-R1). See LICENSE for details.
