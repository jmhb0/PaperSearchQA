"""
python -m ipdb data_gen/make_pubmed_corpus.py
"""
import json
import os
import ipdb
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import chardet
import re


def detect_encoding(file_path: str, sample_size: int = 10000) -> str:
    """Detect the encoding of a file by reading a sample."""
    with open(file_path, 'rb') as f:
        sample = f.read(sample_size)
        result = chardet.detect(sample)
        return result['encoding']


def load_source_data(
        source_file: str = "data/allMeSH_2022.json") -> Dict[str, Any]:
    """
    Load and parse the source allMeSH JSON file with encoding detection.
    
    Args:
        source_file: Path to the source JSON file
    
    Returns:
        Parsed JSON data
    """
    # Check if the source file exists
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file {source_file} does not exist")

    print(f"Processing {source_file}...")

    # Detect file encoding
    print("Detecting file encoding...")
    detected_encoding = detect_encoding(source_file)
    print(f"Detected encoding: {detected_encoding}")

    # Process the JSON file with detected encoding
    encodings_to_try = [
        detected_encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1'
    ]

    data = None
    for encoding in encodings_to_try:
        if encoding is None:
            continue
        try:
            print(f"Trying to read file with encoding: {encoding}")
            with open(source_file, 'r', encoding=encoding,
                      errors='replace') as f:
                data = json.load(f)
            print(f"Successfully loaded file with encoding: {encoding}")
            break
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"Failed with encoding {encoding}: {e}")
            continue

    if data is None:
        raise ValueError(
            "Could not read the file with any of the attempted encodings")

    print(f"Loaded {len(data['articles'])} documents from {source_file}")
    return data


def write_corpus_and_lookup(articles: List[Dict], corpus_file: str,
                            lookup_file: str) -> Tuple[int, int]:
    """
    Write articles to JSONL corpus file and create PMID lookup mapping.
    
    Args:
        articles: List of article dictionaries
        corpus_file: Path for output corpus JSONL file
        lookup_file: Path for output lookup JSON file
    
    Returns:
        Tuple of (number of documents written, number of PMID mappings)
    """
    pmid_to_id = {}

    # Write JSONL corpus file
    with open(corpus_file, 'w', encoding='utf-8') as corpus_f:
        for idx, document in enumerate(
                tqdm(articles, desc="Processing documents")):
            # Extract title and abstract
            title = document.get("title", "")
            abstract = document.get("abstractText", "")
            pmid = document.get("pmid", "")

            # Create formatted contents
            contents = f"TITLE: {title}. ABSTRACT: {abstract}"

            # Create corpus entry
            corpus_entry = {"id": idx, "contents": contents}

            # Write to JSONL file
            corpus_f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')

            # Store PMID mapping
            if pmid:
                pmid_to_id[str(pmid)] = idx

    # Save PMID lookup mapping
    with open(lookup_file, 'w', encoding='utf-8') as lookup_f:
        json.dump(pmid_to_id, lookup_f, indent=2, ensure_ascii=False)

    return len(articles), len(pmid_to_id)


def make_limited_pubmed_corpus_from_dataset(
        dataset_name: str = "jmhb/bioasq_trainv0_n1609") -> Tuple[str, str]:
    """
    Creates a limited JSONL corpus file from a HuggingFace dataset containing PMIDs.
    
    This function:
    1. Loads a HuggingFace dataset with a 'documents' column containing PMID URLs
    2. Extracts PMIDs from URLs like "http://www.ncbi.nlm.nih.gov/pubmed/17064878"
    3. Creates a filtered corpus containing only documents with those PMIDs
    4. Reuses existing helper functions for loading data and writing corpus
    
    Args:
        dataset_name: Name of the HuggingFace dataset to load
    
    Returns:
        tuple: (corpus_file_path, lookup_file_path)
    """
    try:
        from datasets import load_dataset, concatenate_datasets
    except ImportError:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets")

    # Load the dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    print(f"Available splits: {list(dataset.keys())}")

    # Load all splits and add split column
    all_splits = []
    for split_name in dataset.keys():
        split_dataset = dataset[split_name]
        # Add split column to the dataset
        split_dataset = split_dataset.add_column("split", [split_name] *
                                                 len(split_dataset))
        all_splits.append(split_dataset)
        print(
            f"Loaded split '{split_name}' with {len(split_dataset)} examples")

    # Concatenate all splits
    combined_dataset = concatenate_datasets(all_splits)
    print(f"Combined dataset has {len(combined_dataset)} total examples")

    # Add ipdb breakpoint for inspection
    # ipdb.set_trace()

    # Extract PMIDs from the documents column
    pmids_needed = set()

    for example in combined_dataset:
        documents = example.get('documents', [])
        for doc_url in documents:
            # Extract PMID from URL like "http://www.ncbi.nlm.nih.gov/pubmed/17064878"
            match = re.search(r'/pubmed/(\d+)', str(doc_url))
            if match:
                pmids_needed.add(match.group(1))

    print(f"Found {len(pmids_needed)} unique PMIDs in dataset")

    # Load source data using existing helper
    data = load_source_data()

    # Filter documents to only include those with PMIDs in our set
    filtered_articles = []
    for document in data['articles']:
        pmid = str(document.get("pmid", ""))
        if pmid in pmids_needed:
            filtered_articles.append(document)

    print(
        f"Filtered to {len(filtered_articles)} documents matching dataset PMIDs"
    )

    # Create output filenames
    dataset_name_clean = dataset_name.replace('/', '_')
    corpus_file = f"data/pubmed_restricted_{dataset_name_clean}.jsonl"
    lookup_file = f"data/pubmed_restricted_{dataset_name_clean}-lookupPMID.json"

    # Write corpus and lookup using existing helper
    num_docs, num_pmids = write_corpus_and_lookup(filtered_articles,
                                                  corpus_file, lookup_file)

    print(
        f"Created filtered corpus file: {corpus_file} with {num_docs} documents"
    )
    print(f"Created lookup file: {lookup_file} with {num_pmids} PMID mappings")

    return corpus_file, lookup_file


def create_whole_corpus():
    """
    Creates a JSONL corpus file and a PMID lookup mapping from the allMeSH_2022.json file.
    
    This function:
    1. Checks if data/allMeSH_2022.json exists
    2. Iterates over the JSON file to create pubmed.jsonl with sequential IDs and formatted contents
    3. Creates data/pubmed-lookupPMID.json mapping PMIDs to sequential IDs
    """
    # Load source data using helper
    data = load_source_data()

    # Output files
    corpus_file = "data/pubmed.jsonl"
    lookup_file = "data/pubmed-lookupPMID.json"

    # Write corpus and lookup using helper
    num_docs, num_pmids = write_corpus_and_lookup(data['articles'],
                                                  corpus_file, lookup_file)

    print(f"Created corpus file: {corpus_file} with {num_docs} documents")
    print(f"Created lookup file: {lookup_file} with {num_pmids} PMID mappings")


if __name__ == "__main__":
    dataset_name = "jmhb/bioasq_trainv0_n1609"
    dataset_name = "jmhb/bioasq_trainv0_n1609_test100"
    make_limited_pubmed_corpus_from_dataset(dataset_name=dataset_name)
    # create_whole_corpus()
