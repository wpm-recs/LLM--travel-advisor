import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class RetrievalOptimizationModule:
    """Retrieval Optimization Module - Handles hybrid search and filtering logic"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()

    def setup_retrievers(self):
        """Initialize vector-based and BM25-based retrievers"""
        candidate_k = 18

        # Vector-based retriever (Dense Retrieval)
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": candidate_k}
        )

        # MMR retriever increases topical diversity, improving file-level recall.
        self.vector_mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": candidate_k, "fetch_k": candidate_k * 2, "lambda_mult": 0.35}
        )

        # BM25 retriever (Sparse/Keyword Retrieval)
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=candidate_k
        )

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Hybrid Search - Combines vector and BM25 results using Reciprocal Rank Fusion (RRF)
        """
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []

        # Fetch results from both retrievers independently
        vector_docs = self.vector_retriever.get_relevant_documents(normalized_query)
        mmr_docs = self.vector_mmr_retriever.get_relevant_documents(normalized_query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(normalized_query)

        # Re-rank documents using the RRF algorithm
        reranked_docs = self._rrf_rerank(
            {
                "vector": vector_docs,
                "mmr": mmr_docs,
                "bm25": bm25_docs,
            }
        )
        return reranked_docs[:top_k]

    def _doc_key(self, doc: Document) -> str:
        """Generate stable key across retrievers for the same chunk."""
        meta = doc.metadata or {}
        if meta.get("chunk_id"):
            return f"chunk:{meta['chunk_id']}"

        parent = str(meta.get("parent_id", ""))
        index = str(meta.get("chunk_index", ""))
        source = str(meta.get("relative_path", ""))
        content_hint = doc.page_content[:200].strip().lower()
        return f"fallback:{parent}|{index}|{source}|{hash(content_hint)}"

    def _metadata_richness(self, doc: Document) -> int:
        meta = doc.metadata or {}
        score = 0
        for field in ("wiki_title", "wiki_url", "relative_path", "Title", "Section"):
            if meta.get(field):
                score += 1
        return score

    def _rrf_rerank(self, ranked_results: Dict[str, List[Document]]) -> List[Document]:
        """Reciprocal Rank Fusion (RRF) re-ranking algorithm"""

        # RRF scoring dictionary
        rrf_scores = {}
        k = 60  # RRF hyperparameter (smoothing constant)
        all_docs: Dict[str, Document] = {}

        for _, docs in ranked_results.items():
            for rank, doc in enumerate(docs):
                doc_id = self._doc_key(doc)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

                existing = all_docs.get(doc_id)
                if existing is None or self._metadata_richness(doc) > self._metadata_richness(existing):
                    all_docs[doc_id] = doc

        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda key: rrf_scores[key], reverse=True)
        return [all_docs[doc_id] for doc_id in sorted_doc_ids if doc_id in all_docs]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any],
                                 top_k: int = 5) -> List[Document]:
        """Search with metadata pre-filtering applied"""
        # Initialize retriever with metadata filter configuration
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            # Expand initial retrieval range (k*3) to ensure enough candidates after filtering
            search_kwargs={"k": top_k * 3, "filter": filters}
        )

        results = vector_retriever.invoke(query)
        return results[:top_k]