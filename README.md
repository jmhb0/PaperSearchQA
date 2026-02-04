# PaperSearchQA

[![Paper](https://img.shields.io/badge/arXiv-2601.18207-b31b1b)](https://arxiv.org/abs/2601.18207)
[![Website](https://img.shields.io/badge/Website-jmhb0.github.io-blue)](https://jmhb0.github.io/PaperSearchQA/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)]([https://huggingface.co/datasets/jmhb/PaperSearchQA](https://huggingface.co/collections/jmhb/papersearchqa))

**[EACL 2026]** A reinforcement learning environment for training AI agents to autonomously search and reason over scientific literature.

**Paper**: [PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR](https://arxiv.org/abs/2601.18207)

The paper's main contribution is RL training environments: a data generation pipeline for creating Q&As from scientific biomedical paper abstracts; and a retrieval corpus of paper abstracts. We train a search agent in this environment, and show that it generalizes to the [BioASQ](https://huggingface.co/datasets/jmhb/BioASQ) benchmark. The data-generation ideas should generalize to non-biomedical papers as well.

- **`data_generation/`** - Scalable pipeline for generating Q&A pairs from scientific abstracts using GPT-4.1. Includes corpus processing, question generation, golden answer creation, and paraphrasing (~$150-230 for 60K examples).

- **`baselines/`** - Baseline implementations (RAG, CoT, SearchR1) with inference scripts and evaluation code.

- **`search-r1/`** - Retrieval infrastructure from [Search-R1](https://github.com/PeterGriffinJin/Search-R1): BM25/dense retrieval servers, veRL training framework, and data processing utilities.

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/PaperSearchQA.git
cd PaperSearchQA
pip install -r requirements.txt
```

### Load Dataset

The generated data is available on HuggingFace:

```python
from datasets import load_dataset

# Load training and test sets
dataset = load_dataset("jmhb/PaperSearchQA")
```

We also re-release the [BioASQ](http://bioasq.org/) data on HuggingFace, which is a good test of generalization (please cite them if you use it):

```python
# Load BioASQ factoid questions for out-of-distribution evaluation
bioasq = load_dataset("jmhb/BioASQ", "factoid")
```

## Data Generation Pipeline

The main contribution of this work is the scalable pipeline in `data_generation/`:

```bash
cd data_generation

# Optional: Extract LLM response cache (saves API costs)
tar -xzvf cache.tar.gz

# Step 1: Convert allMeSH to parquet (one-time)
python core_pipeline/allMesh_to_parquet.py

# Step 2: Create PubMed corpus (one-time)
python core_pipeline/make_pubmed_corpus.py

# Step 3: Generate Q&A pairs
python core_pipeline/generate_questions_from_abstracts.py
```

**Note**: The `cache.tar.gz` contains cached LLM responses from our data generation run. Extracting it before running Step 3 will significantly reduce API costs by reusing previously generated questions and answers.

See [data_generation/README.md](data_generation/README.md) for full pipeline details.

## Running Baselines

Evaluate the generated dataset with various baseline methods:

```bash
# Start retrieval server (for RAG/SearchR1 methods)
cd search-r1 && bash retrieval_launch_pubmed_bm25.sh && cd ..

# RAG baseline
python baselines/rag/run_inference.py \
    --method rag \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA

# Direct inference or CoT
python baselines/rag/run_inference.py \
    --method direct \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_id PaperSearchQA/PaperSearchQA
```

See [baselines/README.md](baselines/README.md) for all options and [search-r1/README.md](search-r1/README.md) for retrieval setup.

## Dataset Details

The generated dataset includes 20K training and 5K test examples covering gene/protein identification, disease mechanisms, drug information, and biological processes. Questions are paired with multiple acceptable answer variations (synonyms, abbreviations). BioASQ factoid questions are also included for additional evaluation.

Evaluation uses **Exact Match (EM)** with support for multiple golden answers:

```python
from verl.utils.reward_score.qa_em import compute_score_em

golden_answers = {"target": ["dystrophin", "DMD protein"]}
score = compute_score_em("dystrophin", golden_answers)  # 1.0
```

## Citation

If you use PaperSearchQA in your research, please cite:

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

If you use the Search-R1 retrieval infrastructure, also cite:

```bibtex
@article{searchr1,
  title={Search-R1: Train your LLMs to reason and call a search engine with reinforcement learning},
  author={Jin, Peitian and others},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

And if you use the BioASQ datasets then please cite 
```bibtex
@article{krithara2023bioasq,
  title={BioASQ-QA: A manually curated corpus for Biomedical Question Answering},
  author={Krithara, Anastasia and Nentidis, Anastasios and Bougiatiotis, Konstantinos and Paliouras, Georgios},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={170},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

@article{tsatsaronis2015overview,
  title={An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition},
  author={Tsatsaronis, George and Balikas, Georgios and Malakasiotis, Prodromos and Partalas, Ioannis and Zschunke, Matthias and Alvers, Michael R and Weissenborn, Dirk and Krithara, Anastasia and Petridis, Sergios and Polychronopoulos, Dimitris and others},
  journal={BMC bioinformatics},
  volume={16},
  number={1},
  pages={138},
  year={2015},
  publisher={Springer}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details. The PubMed corpus is derived from allMeSH (BioASQ 2022). BioASQ evaluation data requires registration at http://bioasq.org/.

## Acknowledgments

Retrieval infrastructure adapted from [Search-R1](https://github.com/PeterGriffinJin/Search-R1). Corpus based on PubMed abstracts and [BioASQ challenge](https://bioasq.org/) data.


