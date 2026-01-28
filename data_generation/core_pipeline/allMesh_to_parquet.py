"""
python -m ipdb data_gen/allMesh_to_parquet.py
Load the PubMed corpus in a form that's easy to query wrt PMID or row number; and easy to search wrt MESH terms.

Needs `data/allMeSH_2022.json` to exist, downloaded from bioASQ from 2022 challenge https://ceur-ws.org/Vol-3180/paper-10.pdf 

After running, then other files can load it with:

from data_gen.allMesh_to_parquet import return_indexer
indexer = return_indexer('data/allMeSH_2022.parquet')

and then:

doc0 = indexer.loc('12345678')
doc1 = indexer.iloc(0)

"""
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import ijson
import json
import pickle
import os
from typing import Union, List, Dict, Any
import numpy as np
import tqdm
import ipdb


class ArrowPMIDIndexer:
    """Fast Arrow-based indexer with both row number and PMID access"""

    def __init__(self, arrow_file: str):
        """Load Arrow table and build/load PMID index"""
        self.arrow_file = arrow_file
        self.index_file = arrow_file.replace('.parquet', '_pmid_index.pkl')
        self.mesh_index_file = arrow_file.replace('.parquet',
                                                  '_mesh_index.pkl')

        print(f"Loading Arrow table from {arrow_file}")
        self.table = pq.read_table(arrow_file)

        # Try to load existing index, otherwise build it
        if self._load_pmid_index():
            print("Loaded existing PMID index")
        else:
            print("Building new PMID index...")
            self._build_pmid_index()
            self._save_pmid_index()

        # Initialize mesh index as None - built on demand
        self.mesh_to_articles = None

    def _load_pmid_index(self) -> bool:
        """Try to load existing PMID index from disk"""
        if not os.path.exists(self.index_file):
            return False

        try:
            # Check if index file is newer than parquet file
            index_mtime = os.path.getmtime(self.index_file)
            parquet_mtime = os.path.getmtime(self.arrow_file)

            if index_mtime < parquet_mtime:
                print("Index file is older than parquet file, rebuilding...")
                return False

            with open(self.index_file, 'rb') as f:
                self.pmid_to_row = pickle.load(f)
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False

    def _save_pmid_index(self):
        """Save PMID index to disk"""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.pmid_to_row, f)
            print(f"Saved PMID index to {self.index_file}")
        except Exception as e:
            print(f"Failed to save index: {e}")

    def _load_mesh_index(self) -> bool:
        """Try to load existing mesh index from disk"""
        if not os.path.exists(self.mesh_index_file):
            return False

        try:
            # Check if mesh index file is newer than parquet file
            mesh_index_mtime = os.path.getmtime(self.mesh_index_file)
            parquet_mtime = os.path.getmtime(self.arrow_file)

            if mesh_index_mtime < parquet_mtime:
                print(
                    "Mesh index file is older than parquet file, rebuilding..."
                )
                return False

            with open(self.mesh_index_file, 'rb') as f:
                self.mesh_to_articles = pickle.load(f)
            return True
        except Exception as e:
            print(f"Failed to load mesh index: {e}")
            return False

    def _save_mesh_index(self):
        """Save mesh index to disk"""
        try:
            with open(self.mesh_index_file, 'wb') as f:
                pickle.dump(self.mesh_to_articles, f)
            print(f"Saved mesh index to {self.mesh_index_file}")
        except Exception as e:
            print(f"Failed to save mesh index: {e}")

    def _build_mesh_index(self):
        """Build hash map from mesh terms to list of article indices"""
        import time
        from collections import defaultdict

        print("Building mesh term index...")
        start_time = time.time()

        # Initialize the mesh index
        mesh_to_articles = defaultdict(list)

        # Process articles in batches for memory efficiency
        batch_size = 10000
        total_articles = len(self.table)

        for i in tqdm.trange(0,
                             total_articles,
                             batch_size,
                             desc="Building mesh index"):
            end_idx = min(i + batch_size, total_articles)
            batch_articles = self.iloc(list(range(i, end_idx)))

            for article_idx, article in enumerate(batch_articles):
                actual_idx = i + article_idx
                mesh_terms = article.get('meshMajor', [])

                # Add this article index to each mesh term
                for mesh_term in mesh_terms:
                    if mesh_term:  # Skip empty terms
                        mesh_to_articles[mesh_term].append(actual_idx)

        # Convert defaultdict to regular dict for pickling
        self.mesh_to_articles = dict(mesh_to_articles)

        total_time = time.time() - start_time
        print(f"Mesh index build time: {total_time:.2f} seconds")
        print(
            f"Built index for {len(self.mesh_to_articles)} unique mesh terms")

    def _ensure_mesh_index(self):
        """Ensure mesh index is loaded or built"""
        if self.mesh_to_articles is None:
            if self._load_mesh_index():
                print("Loaded existing mesh index")
            else:
                print("Building new mesh index...")
                self._build_mesh_index()
                self._save_mesh_index()

    def get_articles_by_mesh_term(self, mesh_term: str) -> List[int]:
        """
        Get list of article indices that contain the specified mesh term.
        
        Args:
            mesh_term: The mesh term to search for
            
        Returns:
            List of article indices (for use with iloc)
        """
        self._ensure_mesh_index()

        return self.mesh_to_articles.get(mesh_term, [])

    def get_available_mesh_terms(self) -> List[str]:
        """
        Get list of all available mesh terms in the database.
        
        Returns:
            List of all mesh terms
        """
        self._ensure_mesh_index()

        return list(self.mesh_to_articles.keys())

    def search_mesh_terms(self, partial_term: str) -> List[str]:
        """
        Search for mesh terms that contain the given partial term (case-insensitive).
        
        Args:
            partial_term: Partial mesh term to search for
            
        Returns:
            List of matching mesh terms
        """
        self._ensure_mesh_index()

        partial_lower = partial_term.lower()
        matching_terms = []

        for mesh_term in self.mesh_to_articles.keys():
            if partial_lower in mesh_term.lower():
                matching_terms.append(mesh_term)

        return sorted(matching_terms)

    def _build_pmid_index(self):
        """Build hash map from pmid to row number for O(1) lookups"""
        import time
        print("Building PMID index...")
        start_time = time.time()

        print("Extracting PMIDs from Arrow table...")
        pmids = self.table['pmid'].to_pylist()
        extract_time = time.time() - start_time
        print(f"PMID extraction took {extract_time:.2f} seconds")

        print("Building hash map...")
        map_start = time.time()
        self.pmid_to_row = {str(pmid): i for i, pmid in enumerate(pmids)}
        map_time = time.time() - map_start

        total_time = time.time() - start_time
        print(f"Hash map creation took {map_time:.2f} seconds")
        print(f"Total index build time: {total_time:.2f} seconds")
        print(f"Built index for {len(self.pmid_to_row)} PMIDs")

    def iloc(self, row_num: Union[int, slice,
                                  List[int]]) -> Union[Dict, List[Dict]]:
        """Access by row number (like pandas iloc)"""
        if isinstance(row_num, int):
            # Single row
            row = self.table.slice(row_num, 1)
            return row.to_pylist()[0]
        elif isinstance(row_num, slice):
            # Slice of rows
            start, stop, step = row_num.indices(len(self.table))
            if step == 1:
                # Efficient slice
                rows = self.table.slice(start, stop - start)
                return rows.to_pylist()
            else:
                # Handle step != 1 - process individually to avoid Arrow issues
                indices = list(range(start, stop, step))
                result = []
                for idx in indices:
                    row = self.table.slice(idx, 1).to_pylist()[0]
                    result.append(row)
                return result
        elif isinstance(row_num, list):
            # List of row numbers - process individually to avoid Arrow concatenation issues
            result = []
            for idx in row_num:
                row = self.table.slice(idx, 1).to_pylist()[0]
                result.append(row)
            return result

    def loc(
        self, pmid: Union[str, int,
                          List[Union[str, int]]]) -> Union[Dict, List[Dict]]:
        """Access by PMID (like pandas loc)"""
        if isinstance(pmid, (str, int)):
            # Single PMID
            pmid_str = str(pmid)
            if pmid_str not in self.pmid_to_row:
                raise KeyError(f"PMID {pmid} not found")
            row_num = self.pmid_to_row[pmid_str]
            return self.iloc(row_num)
        elif isinstance(pmid, list):
            # List of PMIDs
            row_nums = []
            for p in pmid:
                p_str = str(p)
                if p_str not in self.pmid_to_row:
                    raise KeyError(f"PMID {p} not found")
                row_nums.append(self.pmid_to_row[p_str])
            return self.iloc(row_nums)

    def filter_pmids(self, pmid_list: List[Union[str, int]]) -> List[Dict]:
        """Efficiently filter by multiple PMIDs"""
        pmid_strs = [str(p) for p in pmid_list]
        # Use Arrow's compute functions for efficient filtering
        mask = pc.is_in(self.table['pmid'], value_set=pa.array(pmid_strs))
        filtered_table = self.table.filter(mask)
        return filtered_table.to_pylist()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        """Support indexer[key] syntax"""
        if isinstance(key, (int, slice, list)) and not isinstance(key, str):
            return self.iloc(key)
        else:
            return self.loc(key)


def convert_json_to_arrow(input_file: str, output_file: str):
    """Convert JSON with 'articles' key to Arrow format"""
    print(f"Converting {input_file} to Arrow format...")

    # Try different encodings to handle UTF-8 issues
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']

    articles = None
    for encoding in encodings_to_try:
        try:
            print(f"Trying encoding: {encoding}")
            with open(input_file, 'r', encoding=encoding,
                      errors='replace') as f:
                articles = ijson.items(f, 'articles.item')
                # Test if we can read at least one item
                first_item = next(articles)
                print(f"Successfully opened with {encoding}")

                # Reopen and process all items
                with open(input_file, 'r', encoding=encoding,
                          errors='replace') as f:
                    articles = ijson.items(f, 'articles.item')
                    article_list = list(articles)
                break
        except (UnicodeDecodeError, ijson.common.IncompleteJSONError) as e:
            print(f"Failed with {encoding}: {e}")
            continue

    if articles is None:
        raise ValueError("Could not decode file with any encoding")

    print(f"Loaded {len(article_list)} articles")

    table = pa.Table.from_pylist(article_list)

    # Write to Arrow/Parquet format
    pq.write_table(table, output_file)
    print(f"Saved to {output_file}")


def convert_json_to_arrow_with_index(input_file: str, output_file: str):
    """Convert JSON to Arrow format and prebuild the PMID index"""
    # First convert to parquet
    convert_json_to_arrow(input_file, output_file)

    # Then build and save the index
    print("Building PMID index...")
    indexer = ArrowPMIDIndexer(output_file)
    print("Index building complete!")
    return indexer
    """Create and return an ArrowPMIDIndexer for the given parquet file"""
    return ArrowPMIDIndexer(parquet_file)


def return_indexer(parquet_file: str) -> ArrowPMIDIndexer:
    """Create and return an ArrowPMIDIndexer for the given parquet file"""
    return ArrowPMIDIndexer(parquet_file)


def extract_unique_mesh_terms(indexer: ArrowPMIDIndexer) -> set:
    """
    Scan the entire database and extract all unique MESH terms.
    
    Args:
        indexer: ArrowPMIDIndexer object containing the PubMed data
        
    Returns:
        Set of unique MESH terms found in the database
    """
    total_articles = len(indexer)
    unique_mesh_terms = set()
    batch_size = 10000

    print("Extracting MESH terms...")
    for i in tqdm.trange(0, total_articles, batch_size):
        # Get batch of articles
        end_idx = min(i + batch_size, total_articles)
        batch_articles = indexer.iloc(list(range(i, end_idx)))

        # Extract mesh terms from each article in the batch
        for article in batch_articles:
            mesh_terms = article.get('meshMajor', [])
            unique_mesh_terms.update(mesh_terms)

    print(f"Found {len(unique_mesh_terms)} unique MESH terms")
    return unique_mesh_terms


def save_mesh_terms_to_file(mesh_terms: set,
                            output_file: str = "data/unique_mesh_terms.txt"):
    """
    Save the unique MESH terms to a text file.
    
    Args:
        mesh_terms: Set of unique MESH terms
        output_file: Path to save the MESH terms
    """
    print(f"Saving {len(mesh_terms)} unique MESH terms to {output_file}")
    sorted_terms = sorted(mesh_terms)
    with open(output_file, 'w', encoding='utf-8') as f:
        for term in sorted_terms:
            f.write(f"{term}\n")


def get_mesh_terms_summary(mesh_terms: set) -> dict:
    """
    Get a summary of the MESH terms data.
    
    Args:
        mesh_terms: Set of unique MESH terms
        
    Returns:
        Dictionary with summary statistics
    """
    if not mesh_terms:
        return {"total_count": 0, "sample_terms": []}

    sorted_terms = sorted(mesh_terms)

    return {
        "total_count": len(mesh_terms),
        "sample_terms": sorted_terms[:10],  # First 10 terms alphabetically
        "avg_term_length":
        sum(len(term) for term in mesh_terms) / len(mesh_terms),
        "min_length": min(len(term) for term in mesh_terms),
        "max_length": max(len(term) for term in mesh_terms),
        "longest_term": max(mesh_terms, key=len),
        "shortest_term": min(mesh_terms, key=len)
    }


def get_article_pool(indexer, leaf_map):
    """
    Create a dict mapping each unique drug to a list of pubmed idxs from indexer.
    
    Args:
        indexer: ArrowPMIDIndexer object with get_articles_by_mesh_term method
        leaf_map: Dict mapping drug categories to dicts of {drug_uid: drug_name}
    
    Returns:
        Dict mapping drug_name to list of article indices
    """
    drugs_to_sampleidx = {}

    # Collect all unique drugs from all categories
    all_drugs = set()
    for category_drugs in leaf_map.values():
        for drug_uid, drug_name in category_drugs.items():
            # Assert that drug_uid looks like a MeSH term (starts with D and has numbers)
            assert drug_uid.startswith('D') and drug_uid[1:].isdigit(), \
                f"Drug UID {drug_uid} doesn't look like a MeSH term"
            all_drugs.add(drug_name)

    print(f"Found {len(all_drugs)} unique drugs across all categories")

    # For each unique drug, get the article indices
    for drug_name in tqdm.tqdm(all_drugs, desc="Getting articles for drugs"):
        article_indices = indexer.get_articles_by_mesh_term(drug_name)
        drugs_to_sampleidx[drug_name] = article_indices

    return drugs_to_sampleidx


# Usage example
if __name__ == "__main__":
    # Step 1: Convert JSON to Arrow AND build index (one-time setup)
    # indexer = convert_json_to_arrow_with_index('data/allMeSH_2022.json',
    #                                            'data/allMeSH_2022.parquet')

    # Step 2: Load the indexer (fast!)
    print("Loading indexer...")
    indexer = return_indexer('data/allMeSH_2022.parquet')

    # Step 4: Demonstrate mesh term search functionality
    print("\n" + "=" * 50)
    print("MESH TERM SEARCH EXAMPLES")
    print("=" * 50)

    # Example 1: Search for articles with a specific mesh term
    example_mesh_term = "Pharmaceutical Preparations"
    print(f"\nSearching for articles with mesh term: '{example_mesh_term}'")
    article_indices = indexer.get_articles_by_mesh_term(example_mesh_term)
    print(f"Found {len(article_indices)} articles")

    if article_indices:
        print(f"First 5 article indices: {article_indices[:5]}")

        # Get the actual articles using iloc
        sample_articles = indexer.iloc(
            article_indices[:3])  # Get first 3 articles
        print(f"\nSample articles with '{example_mesh_term}':")
        for i, article in enumerate(sample_articles):
            print(f"  Article {i+1}: PMID {article.get('pmid', 'N/A')}")
            print(f"    Title: {article.get('title', 'N/A')[:100]}...")
            print(f"    Mesh terms: {article.get('meshMajor', [])[:3]}..."
                  )  # Show first 3 mesh terms

    # Example 2: Search for mesh terms containing a keyword
    print(f"\n" + "-" * 30)
    print("MESH TERM KEYWORD SEARCH")
    print("-" * 30)

    keyword = "Drug"
    matching_terms = indexer.search_mesh_terms(keyword)
    print(f"Mesh terms containing '{keyword}': {len(matching_terms)}")
    print(f"First 10 matches: {matching_terms[:10]}")

    # Example 3: Show some available mesh terms
    print(f"\n" + "-" * 30)
    print("AVAILABLE MESH TERMS SAMPLE")
    print("-" * 30)

    all_mesh_terms = indexer.get_available_mesh_terms()
    print(f"Total mesh terms available: {len(all_mesh_terms)}")
    print(f"Sample mesh terms: {all_mesh_terms[:10]}")

    ipdb.set_trace()

    # Usage examples:

    # Access by row number (like iloc)
    article_0 = indexer.iloc(0)  # First article
    articles_slice = indexer.iloc(slice(100, 200))  # Articles 100-199
    specific_rows = indexer.iloc([0, 50, 100])  # Specific row numbers

    # Access by PMID (like loc)
    article_by_pmid = indexer.loc('34823483')  # Single PMID
    articles_by_pmids = indexer.loc(['34823483', '34821622'])  # Multiple PMIDs

    # Alternative syntax
    article = indexer[0]  # Same as iloc(0)
    article = indexer['34823483']  # Same as loc('34823483')

    # Efficient filtering
    filtered = indexer.filter_pmids(['34823483', '34821622'])

    print(f"Total articles: {len(indexer)}")
    print(f"Sample pmid: {article_by_pmid.get('pmid', 'Not found')}")
