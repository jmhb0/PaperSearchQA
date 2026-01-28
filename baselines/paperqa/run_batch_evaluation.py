# run_batch_evaluation.py
# Batch evaluation script for PaperQA with 100 questions from PaperSearchQA dataset
# Tested with: paper-qa==4.4.0, langchain-core>=0.2.4, httpx, openai>=1.0.0
# env:
#   Start vLLM server: vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8003
#   export RETRIEVE_URL="http://localhost:8001/retrieve"
#   export RETRIEVE_BEARER_TOKEN="..."            # optional
#   export MAX_FETCH=50                           # optional cap


from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
import asyncio, os, httpx, json, hashlib
from datasets import load_dataset
from tqdm import tqdm
import time
import diskcache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from paperqa import Docs, LangchainVectorStore as _LCVS
from paperqa.types import Text, PromptCollection
from openai import AsyncOpenAI

MAX_SOURCES = 3  # Limit number of document chunks for processing

# Dataset configuration
DATASET_NAME = "PaperSearchQA/BioASQ_factoid"  # Dataset to evaluate on
N_SAMPLES = 250  # Number of questions to process from dataset

# Concurrency configuration
MAX_CONCURRENT_WORKERS = 50  # Number of questions to process simultaneously

# Cache configuration
ENABLE_CACHE = True                 # Set to False to disable caching
FORCE_REFRESH = False               # Set to True to ignore existing answer cache and reprocess all questions
ENABLE_RETRIEVAL_CACHE = True       # Set to False to disable retrieval caching
FORCE_REFRESH_RETRIEVAL = False     # Set to True to ignore existing retrieval cache and re-fetch all documents
CACHE_DIR = "./paperqa_cache"
RETRIEVAL_CACHE_DIR = "./paperqa_cache/retrieval"

# Initialize caches (will be updated in main() with dataset-specific paths)
cache = None
retrieval_cache = None

# -------------------------------
# Simple wrapper to make OpenAI client compatible with LangchainLLMModel
# -------------------------------
class OpenAIClientWrapper:
    """Wrapper to make OpenAI client compatible with PaperQA's LangchainLLMModel."""
    def __init__(self, openai_client: AsyncOpenAI, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self._client = openai_client
        self.model_name = model_name

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def ainvoke(self, messages):
        """LangChain-style completion interface for chat models."""
        if isinstance(messages, str):
            # Completion mode
            response = await self._client.completions.create(
                model=self.model_name,
                prompt=messages,
                temperature=0.1,
                max_tokens=3000,
                stop=["</answer>", "<|im_end|>", "<|endoftext|>"]
            )
            return response.choices[0].text
        else:
            # Chat mode
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=3000,
                stop=["</answer>", "<|im_end|>", "<|endoftext|>"]
            )
            return response.choices[0].message

    async def astream(self, messages):
        """LangChain-style streaming interface."""
        if isinstance(messages, str):
            # Completion mode
            response = await self._client.completions.create(
                model=self.model_name,
                prompt=messages,
                temperature=0.1,
                max_tokens=3000,
                stream=True,
                stop=["</answer>", "<|im_end|>", "<|endoftext|>"]
            )
            async for chunk in response:
                if chunk.choices[0].text:
                    yield chunk.choices[0].text
        else:
            # Chat mode
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=3000,
                stream=True,
                stop=["</answer>", "<|im_end|>", "<|endoftext|>"]
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


# -------------------------------
# HTTP-backed retriever (LangChain VectorStore)
# -------------------------------
class HTTPVectorStore(VectorStore):
    """Retrieval-only VectorStore that calls your FastAPI /retrieve endpoint."""
    def __init__(self, base_url: str, timeout: float = 30.0, headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}

    # ---- ABC stubs (we don't build an index here) ----
    @classmethod
    def from_texts(cls, texts: List[str], embedding: Optional[Embeddings] = None,
                   metadatas: Optional[List[Dict[str, Any]]] = None, **kwargs: Any):
        raise NotImplementedError
    @classmethod
    def from_documents(cls, documents: List[Document], embedding: Optional[Embeddings] = None, **kwargs: Any):
        raise NotImplementedError
    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[Dict[str, Any]]] = None, **kwargs: Any):
        raise NotImplementedError
    def add_documents(self, documents: List[Document], **kwargs: Any):
        raise NotImplementedError

    # ---- Response normalization for your server ----
    @staticmethod
    def _extract_hits(res: Any) -> List[Any]:
        """
        Your server: res["result"][0] is the list of hits
        Accept fallbacks: {"results":[...]}, {"hits":[...]}, [[...]], [...]
        """
        if isinstance(res, dict) and "result" in res and isinstance(res["result"], list):
            first = res["result"][0] if res["result"] else []
            return first if isinstance(first, list) else []
        if isinstance(res, dict):
            return res.get("results") or res.get("hits") or []
        if isinstance(res, list):
            return res[0] if (res and isinstance(res[0], list)) else res
        return []

    @staticmethod
    def _safe_str(x: Any) -> str:
        return "" if x is None else str(x)

    @staticmethod
    def _coerce_hit(hit: Any) -> Tuple[Document, float]:
        """
        Map to Text schema via Document.metadata:
          {"text": <str>, "name": <str>, "doc": {"dockey": <str>, "docname": <str>, ...}}
        """
        s = 0.0
        # Your primary shape: {"id": "...", "contents": "...", (optional) "score","title","citation","url","name"}
        if isinstance(hit, dict) and ("contents" in hit or "id" in hit):
            txt   = HTTPVectorStore._safe_str(hit.get("contents", ""))
            hid   = HTTPVectorStore._safe_str(hit.get("id", ""))
            title = HTTPVectorStore._safe_str(hit.get("title", hid))
            name  = HTTPVectorStore._safe_str(hit.get("name", f"chunk-{hid}"))
            doc_meta = {"dockey": hid, "docname": title}
            # Citation is required by Text class, provide default if missing
            doc_meta["citation"] = HTTPVectorStore._safe_str(hit.get("citation", f"Document {hid}"))
            if hit.get("url"):      doc_meta["url"]      = HTTPVectorStore._safe_str(hit["url"])
            try:
                s = float(hit.get("score", 0.0))
            except Exception:
                s = 0.0
            meta = {"text": txt, "name": name, "doc": doc_meta}
            return Document(page_content=txt, metadata=meta), s

        # Tolerate: bare string
        if isinstance(hit, str):
            txt = hit
            meta = {"text": txt, "name": "chunk", "doc": {"dockey": "unknown", "docname": "unknown", "citation": "Unknown Document"}}
            return Document(page_content=txt, metadata=meta), 0.0

        # Tolerate: [text, score?]
        if isinstance(hit, (list, tuple)):
            txt = "" if not hit else str(hit[0])
            try:
                s = float(hit[1]) if len(hit) > 1 else 0.0
            except Exception:
                s = 0.0
            meta = {"text": txt, "name": "chunk", "doc": {"dockey": "unknown", "docname": "unknown", "citation": "Unknown Document"}}
            return Document(page_content=txt, metadata=meta), s

        # Tolerate: dicts with text/metadata-ish keys
        if isinstance(hit, dict):
            txt  = str(hit.get("text") or hit.get("content") or hit.get("page_content") or "")
            meta_in = hit.get("metadata") or hit.get("meta") or {}
            name = HTTPVectorStore._safe_str(meta_in.get("name", "chunk"))
            dockey  = HTTPVectorStore._safe_str(meta_in.get("doc_id") or meta_in.get("dockey") or "unknown")
            docname = HTTPVectorStore._safe_str(meta_in.get("title")  or meta_in.get("docname") or dockey)
            doc_meta = {"dockey": dockey, "docname": docname}
            # Citation is required by Text class, provide default if missing
            doc_meta["citation"] = HTTPVectorStore._safe_str(meta_in.get("citation", f"Document {dockey}"))
            if "url" in meta_in:      doc_meta["url"]      = HTTPVectorStore._safe_str(meta_in["url"])
            try:
                s = float(hit.get("score", meta_in.get("score", 0.0)))
            except Exception:
                s = 0.0
            meta = {"text": txt, "name": name, "doc": doc_meta}
            return Document(page_content=txt, metadata=meta), s

        # Fallback
        txt = str(hit)
        meta = {"text": txt, "name": "chunk", "doc": {"dockey": "unknown", "docname": "unknown", "citation": "Unknown Document"}}
        return Document(page_content=txt, metadata=meta), 0.0

    # ---- Core search methods PaperQA calls ----
    async def asimilarity_search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        k = min(k, int(os.getenv("MAX_FETCH", "50")))

        # Check retrieval cache first (only if enabled and not forcing refresh)
        if ENABLE_RETRIEVAL_CACHE and retrieval_cache is not None and not FORCE_REFRESH_RETRIEVAL:
            cache_key = hashlib.sha256(f"search_{query}_{k}_retrieval".encode()).hexdigest()
            cached_docs = retrieval_cache.get(cache_key)
            if cached_docs is not None:
                # print(f"[retrieval] cache hit for query: {query[:50]}...")
                # Reconstruct Document objects from cached data
                return [Document(page_content=doc_data["page_content"], metadata=doc_data["metadata"])
                       for doc_data in cached_docs]

        # Perform actual retrieval
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            r = await client.post(self.base_url, json={"queries": [query], "topk": k})
            r.raise_for_status()
            res = r.json()
        hits = self._extract_hits(res)
        documents = [self._coerce_hit(h)[0] for h in hits]

        # Cache the results
        if ENABLE_RETRIEVAL_CACHE and retrieval_cache is not None:
            cache_key = hashlib.sha256(f"search_{query}_{k}_retrieval".encode()).hexdigest()
            # Store serializable document data
            doc_data = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
            retrieval_cache.set(cache_key, doc_data, expire=60*60*24*7)  # 7 day expiration

        # print(f"[adapter] parsed {len(documents)} hits")  # uncomment to debug
        return documents

    def similarity_search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Use `await asimilarity_search(...)` inside async contexts.")
        except RuntimeError:
            pass
        return asyncio.run(self.asimilarity_search(query, k=k, **kwargs))

    async def asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        k = min(k, int(os.getenv("MAX_FETCH", "50")))

        # Check retrieval cache first (only if enabled and not forcing refresh)
        if ENABLE_RETRIEVAL_CACHE and retrieval_cache is not None and not FORCE_REFRESH_RETRIEVAL:
            cache_key = hashlib.sha256(f"search_scores_{query}_{k}_retrieval".encode()).hexdigest()
            cached_results = retrieval_cache.get(cache_key)
            if cached_results is not None:
                # print(f"[retrieval] cache hit for scored query: {query[:50]}...")
                # Reconstruct Document objects and scores from cached data
                return [(Document(page_content=item["doc"]["page_content"], metadata=item["doc"]["metadata"]), item["score"])
                       for item in cached_results]

        # Perform actual retrieval
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            r = await client.post(self.base_url, json={"queries": [query], "topk": k})
            r.raise_for_status()
            res = r.json()
        hits = self._extract_hits(res)
        results = [self._coerce_hit(h) for h in hits]

        # Cache the results
        if ENABLE_RETRIEVAL_CACHE and retrieval_cache is not None:
            cache_key = hashlib.sha256(f"search_scores_{query}_{k}_retrieval".encode()).hexdigest()
            # Store serializable document and score data
            cache_data = [{"doc": {"page_content": doc.page_content, "metadata": doc.metadata}, "score": score}
                         for doc, score in results]
            retrieval_cache.set(cache_key, cache_data, expire=60*60*24*7)  # 7 day expiration

        return results

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        docs = self.similarity_search(query, k=k, **kwargs)
        return [(d, float(d.metadata.get("score", 0.0))) for d in docs]

    def max_marginal_relevance_search(
        self, query: str, k: int = 10, fetch_k: int | None = None, lambda_mult: float = 0.5, **kwargs: Any
    ) -> List[Document]:
        size = min(fetch_k or k, int(os.getenv("MAX_FETCH", "50")))
        docs = self.similarity_search(query, k=size, **kwargs)
        return docs[:k]


# -------------------------------
# Disable MMR inside PaperQA (no embedding math)
# -------------------------------
class NoMMRVectorStore(_LCVS):
    async def max_marginal_relevance_search(
        self,
        client,
        query: str,
        k: int,
        fetch_k: int | None = None,
        lambda_mult: float = 0.5,
        **kwargs,
    ):
        # Plain similarity; skip cosine/MMR entirely
        size = fetch_k or k
        texts, scores = await self.similarity_search(client, query, size)

        # Trim and keep internal caches consistent
        texts = texts[:k]
        scores = scores[:k]
        self._texts = texts
        self._scores = scores

        # Return what PaperQA expects
        return texts, scores


def extract_answer_from_response(response_text: str) -> str:
    """Extract text between <answer> and </answer> tags, or return full response if no tags."""
    if "<answer>" in response_text:
        start = response_text.find("<answer>") + len("<answer>")
        end = response_text.find("</answer>", start)
        if end != -1:
            return response_text[start:end].strip()
        else:
            # Handle case where </answer> was cut off by stop token
            return response_text[start:].strip()
    return response_text.strip()


def generate_cache_key(question: str, model_name: str, max_sources: int, retrieve_url: str, prompt_hash: str) -> str:
    """Generate a unique cache key based on all relevant parameters."""
    # Combine all parameters that affect the result
    key_components = {
        "question": question,
        "model_name": model_name,  # This is what really matters for LLM responses
        "max_sources": max_sources,
        "prompt_hash": prompt_hash,
        "paperqa_version": "4.4.0",  # Include version for compatibility
        "output_format_version": "v2"  # Change this when output format changes
    }

    # Create a deterministic hash
    key_string = json.dumps(key_components, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()


def get_prompt_hash(prompts: PromptCollection) -> str:
    """Generate a hash of the prompt configuration."""
    # Hash the QA prompt since that's what affects final output
    return hashlib.sha256(prompts.qa.encode()).hexdigest()[:16]


async def process_question(docs: Docs, question: str, question_id: str, model_name: str, retrieve_url: str, prompt_hash: str, dataset_index: int, total_questions: int) -> Dict[str, Any]:
    """Process a single question and return results."""
    start_time = time.time()

    # Check cache first (only if enabled and not forcing refresh)
    if ENABLE_CACHE and cache is not None and not FORCE_REFRESH:
        cache_key = generate_cache_key(question, model_name, MAX_SOURCES, retrieve_url, prompt_hash)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            # Handle both old and new cache formats
            preview = cached_result.get('answer_preview') or cached_result.get('extracted_answer', 'N/A')
            preview_truncated = preview[:100] + "..." if len(preview) > 100 else preview
            print(f"[{question_id}] 💾 Cache hit! ({dataset_index}/{total_questions}) Preview: {preview_truncated}")
            cached_result["question_id"] = question_id  # Update question_id for current run
            cached_result["cached"] = True
            return cached_result

    # Show that we're starting to process this question
    print(f"[{question_id}] 🔄 Processing: {question[:80]}...")

    try:
        result = await docs.aquery(question, max_sources=MAX_SOURCES)
        end_time = time.time()

        # Save the full raw response
        raw_answer = result.formatted_answer

        # Extract answer for preview only (you can apply your own logic later)
        extracted_answer = extract_answer_from_response(raw_answer)

        # Show preview in console (truncated)
        preview_answer = extracted_answer[:100] + "..." if len(extracted_answer) > 100 else extracted_answer
        print(f"[{question_id}] ✓ Completed! ({dataset_index}/{total_questions}) Preview: {preview_answer} (Time: {end_time - start_time:.2f}s)")

        result_data = {
            "question_id": question_id,
            "question": question,
            "full_response": raw_answer,  # Complete PaperQA response with citations
            "answer_preview": extracted_answer,  # Basic extraction for quick viewing
            "contexts": [{"text": ctx.text.text, "score": ctx.score, "citation": ctx.text.doc.citation} for ctx in result.contexts],  # Full context data
            "processing_time": end_time - start_time,
            "success": True,
            "error": None,
            "cached": False
        }

        # Cache the successful result (only if cache is enabled)
        if ENABLE_CACHE and cache is not None:
            cache_key = generate_cache_key(question, model_name, MAX_SOURCES, retrieve_url, prompt_hash)
            cache.set(cache_key, result_data, expire=60*60*24*30)  # 30 day expiration

        return result_data

    except Exception as e:
        end_time = time.time()
        print(f"[{question_id}] ✗ Failed: ({dataset_index}/{total_questions}) {str(e)[:100]}... (Time: {end_time - start_time:.2f}s)")

        error_result = {
            "question_id": question_id,
            "question": question,
            "raw_answer": None,
            "extracted_answer": None,
            "processing_time": end_time - start_time,
            "success": False,
            "error": str(e),
            "cached": False
        }
        # IMPORTANT: We explicitly DO NOT cache errors
        # This ensures failed requests are retried on subsequent runs
        return error_result


# -------------------------------
# Main evaluation script
# -------------------------------
async def main():
    global cache, retrieval_cache

    print(f"Loading {DATASET_NAME} dataset...")

    # Load the dataset - try test split first, then train
    try:
        try:
            dataset = load_dataset(DATASET_NAME, split="test")
            split_used = "test"
        except ValueError:
            # If test split doesn't exist, try train split
            dataset = load_dataset(DATASET_NAME, split="train")
            split_used = "train"

        print(f"Loaded {len(dataset)} questions from {split_used} set")
        print(f"Sample fields: {list(dataset[0].keys())}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Take first N questions for experimentation
    questions = dataset[:N_SAMPLES]

    # Setup dataset-specific cache directories
    dataset_safe_name = DATASET_NAME.replace("/", "_").replace("-", "_")
    dataset_cache_dir = f"{CACHE_DIR}/{dataset_safe_name}"
    dataset_retrieval_cache_dir = f"{RETRIEVAL_CACHE_DIR}/{dataset_safe_name}"

    # Initialize caches with dataset-specific paths
    cache = diskcache.FanoutCache(dataset_cache_dir, shards=4, timeout=1) if ENABLE_CACHE else None
    retrieval_cache = diskcache.FanoutCache(dataset_retrieval_cache_dir, shards=4, timeout=1) if ENABLE_RETRIEVAL_CACHE else None

    print("Setting up PaperQA...")

    # Endpoint & headers
    url = os.getenv("RETRIEVE_URL", "http://localhost:8001/retrieve")
    headers: Dict[str, str] = {}
    tok = os.getenv("RETRIEVE_BEARER_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    # Build your HTTP store
    http_store = HTTPVectorStore(url, headers=headers)

    # Two-arg builder PaperQA expects; returns our store instance
    def store_builder(_texts, _embeddings):
        return http_store

    # Use the NoMMR wrapper and build Text objects (CRITICAL)
    texts_index = NoMMRVectorStore(
        store_builder=store_builder,
        class_type=Text,     # <-- build Texts, not Embeddables
        mmr_lambda=0.0,
    )

    # Belt + suspenders: pin the store so JIT can't swap it
    texts_index._store = http_store
    texts_index._store_builder = store_builder

    # LLM client for local vLLM server with Qwen2.5-7B-Instruct
    openai_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8003/v1"
    )

    # Wrap it to be compatible with LangchainLLMModel
    client = OpenAIClientWrapper(openai_client)

    # Modify prompts to include answer tag formatting
    custom_prompts = PromptCollection()
    custom_prompts.qa = custom_prompts.qa + "\n\nIMPORTANT: You can provide detailed reasoning and explanation, but always end your response with <answer></answer> tags containing ONLY the single entity, fact, or short phrase that directly answers the question. The content between the answer tags should be concise (2-4 words maximum)."

    # IMPORTANT: prevent PaperQA from rebuilding/clearing the store
    # Use "langchain" as llm type and pass OpenAI client as langchain client
    docs = Docs(
        llm="langchain",
        client=client,
        texts_index=texts_index,
        jit_texts_index=False,
        prompts=custom_prompts
    )

    print(f"Starting concurrent evaluation of {len(questions['question'])} questions...")

    # Get parameters for cache key
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    retrieve_url = url
    prompt_hash = get_prompt_hash(custom_prompts)

    print(f"Cache configuration:")
    print(f"  Answer cache enabled: {ENABLE_CACHE}")
    print(f"  Retrieval cache enabled: {ENABLE_RETRIEVAL_CACHE}")
    print(f"  Force refresh answers: {FORCE_REFRESH}")
    print(f"  Force refresh retrieval: {FORCE_REFRESH_RETRIEVAL}")
    print(f"  Answer cache directory: {dataset_cache_dir}")
    print(f"  Retrieval cache directory: {dataset_retrieval_cache_dir}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Model: {model_name}")
    print(f"  Max sources: {MAX_SOURCES}")
    print(f"  Prompt hash: {prompt_hash}")

    # Create all tasks for concurrent processing
    tasks = []
    total_questions = len(questions['question'])
    for i, question in enumerate(questions['question']):
        question_id = f"q_{i+1:03d}"
        dataset_index = i + 1  # 1-based indexing for user display
        tasks.append(process_question(docs, question, question_id, model_name, retrieve_url, prompt_hash, dataset_index, total_questions))

    print(f"\nStarting processing of {len(tasks)} questions with max {MAX_CONCURRENT_WORKERS} concurrent workers...")
    print("=" * 80)

    # Process with controlled concurrency using semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

    async def bounded_task(task, task_num):
        async with semaphore:
            return await task

    bounded_tasks = [bounded_task(task, i+1) for i, task in enumerate(tasks)]
    results = await asyncio.gather(*bounded_tasks)

    # Save results with dataset name in filename
    dataset_safe_name = DATASET_NAME.replace("/", "_").replace("-", "_")
    output_file = f"paperqa_{dataset_safe_name}_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== EVALUATION COMPLETE ===")
    print(f"Results saved to: {output_file}")

    # Print summary statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        avg_time = sum(r["processing_time"] for r in successful) / len(successful)
        print(f"Average processing time: {avg_time:.2f}s")

        total_time = sum(r["processing_time"] for r in results)
        print(f"Total processing time: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    if failed:
        print(f"\nFailed questions:")
        for r in failed:
            print(f"  {r['question_id']}: {r['error']}")


if __name__ == "__main__":
    asyncio.run(main())
