# run_retrieval_ollama.py
# Tested with: paper-qa==4.4.0, langchain-core>=0.2.4, httpx, requests>=2.0.0
# env:
#   Start Ollama: ./ollama serve (in background)
#   Download model: ./ollama pull qwen2.5:3b
#   export RETRIEVE_URL="http://localhost:8001/retrieve"
#   export RETRIEVE_BEARER_TOKEN="..."            # optional
#   export MAX_FETCH=50                           # optional cap

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
import asyncio, os, httpx, json

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from paperqa import Docs, LangchainVectorStore as _LCVS
from paperqa.types import Text, PromptCollection

TIMEOUT=150
# -------------------------------
# Simple Ollama client wrapper for PaperQA compatibility
# -------------------------------
class OllamaClientWrapper:
    """Wrapper to make Ollama API compatible with PaperQA's LangchainLLMModel."""
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "qwen2.5:3b"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    async def ainvoke(self, messages):
        """LangChain-style completion interface for chat models."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if isinstance(messages, str):
                # Completion mode
                payload = {
                    "model": self.model_name,
                    "prompt": messages,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 3000}
                }
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
            else:
                # Chat mode
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 3000}
                }
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                result = response.json()
                # Return a message-like object
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                return MockMessage(result.get("message", {}).get("content", ""))

    async def astream(self, messages):
        """LangChain-style streaming interface."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if isinstance(messages, str):
                # Completion mode
                payload = {
                    "model": self.model_name,
                    "prompt": messages,
                    "stream": True,
                    "options": {"temperature": 0.1, "num_predict": 3000}
                }
                async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data and data["response"]:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue
            else:
                # Chat mode
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.1, "num_predict": 3000}
                }
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    content = data["message"]["content"]
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue


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
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            r = await client.post(self.base_url, json={"queries": [query], "topk": k})
            r.raise_for_status()
            res = r.json()
        hits = self._extract_hits(res)
        # print(f"[adapter] parsed {len(hits)} hits")  # uncomment to debug
        return [self._coerce_hit(h)[0] for h in hits]

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
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            r = await client.post(self.base_url, json={"queries": [query], "topk": k})
            r.raise_for_status()
            res = r.json()
        hits = self._extract_hits(res)
        return [self._coerce_hit(h) for h in hits]

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


# -------------------------------
# Wire into PaperQA (force usage)
# -------------------------------
if __name__ == "__main__":
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

    # LLM client for local Ollama server with Qwen2.5-3B
    client = OllamaClientWrapper(
        base_url="http://localhost:8002",
        model_name="qwen2.5:3b"
    )

    # Modify prompts to include answer tag formatting
    custom_prompts = PromptCollection()
    custom_prompts.qa = custom_prompts.qa + "\n\nIMPORTANT: You can provide detailed reasoning and explanation, but always end your response with <answer></answer> tags containing ONLY the single entity, fact, or short phrase that directly answers the question. The content between the answer tags should be concise (2-4 words maximum)."

    # IMPORTANT: prevent PaperQA from rebuilding/clearing the store
    # Use "langchain" as llm type and pass Ollama client as langchain client
    docs = Docs(
        llm="langchain",
        client=client,
        texts_index=texts_index,
        jit_texts_index=False,
        prompts=custom_prompts
    )

    # Try a query (no need for formatting instructions in query since it's in the system prompt now)
    q = "What anatomical structure, when atretic congenitally, can lead to unilateral hydrocephalus?"
    ans = docs.query(q)
    print("\n=== ANSWER ===\n", ans.formatted_answer)
