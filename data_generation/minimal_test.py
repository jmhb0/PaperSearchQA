#!/usr/bin/env python3
"""
Minimal test - just 5 abstracts to verify pipeline works quickly.
Run this before the full smoke test.
"""

import os
import sys
from pathlib import Path

# Configure for minimal test
os.environ["ENABLE_HF_UPLOAD"] = "0"  # Don't upload for minimal test

# Add core_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "core_pipeline"))

from huggingface_hub import hf_hub_download
from core_pipeline.generate_questions_from_abstracts import generate_dataset_from_abstracts

print("=" * 80)
print("MINIMAL PIPELINE TEST - 5 abstracts (~15 Q&A pairs)")
print("=" * 80)
print("\nThis test verifies the pipeline works end-to-end.")
print("It will cost ~$0.50 and take ~2-3 minutes (plus download time).\n")

# Check if data file exists, download if not
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)
parquet_file = data_dir / "allMeSH_2022.parquet"

if not parquet_file.exists():
    print("Downloading PubMed corpus from HuggingFace (13GB)...")
    print("This is a one-time download that will be reused.\n")

    downloaded_path = hf_hub_download(
        repo_id="jmhb/pubmed_bioasq_2022",
        filename="data/allMeSH_2022.parquet",  # Fixed: file is in data/ subdirectory
        repo_type="dataset",
        local_dir=str(Path(__file__).parent),
        local_dir_use_symlinks=False
    )
    print(f"✅ Download complete: {downloaded_path}\n")
else:
    print(f"✅ Using existing data file: {parquet_file}\n")

# Run with minimal parameters
generate_dataset_from_abstracts(
    key=5,  # Template 5 (3 Q&A per abstract)
    golden_key=3,  # Golden answers template
    n_samples=5,  # Just 5 abstracts
    n_test=2,  # 2 test samples
    hub_name="minimal-test",
    do_paraphrase=False,  # Skip paraphrasing for speed
    paraphrase_key=1,
    paraphrase_pcnt=0
)

print("\n" + "=" * 80)
print("✅ Minimal test complete!")
print("=" * 80)
print("\nNext steps:")
print("1. Review output in output/jmhb_minimal-test_*.parquet")
print("2. Run full smoke test: python smoke_test.py")
