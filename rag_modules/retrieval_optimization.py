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
        # Vector-based retriever (Dense Retrieval)
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25 retriever (Sparse/Keyword Retrieval)
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Hybrid Search - Combines vector and BM25 results using Reciprocal Rank Fusion (RRF)
        """
        # Fetch results from both retrievers independently
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)

        # Re-rank documents using the RRF algorithm
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def _rrf_rerank(self, vector_results: List[Document], bm25_results: List[Document]) -> List[Document]:
        """Reciprocal Rank Fusion (RRF) re-ranking algorithm"""

        # RRF scoring dictionary
        rrf_scores = {}
        k = 60  # RRF hyperparameter (smoothing constant)

        # Calculate RRF scores for vector search results
        for rank, doc in enumerate(vector_results):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # Calculate RRF scores for BM25 search results
        for rank, doc in enumerate(bm25_results):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # Merge all documents and sort them by their calculated RRF scores
        all_docs = {id(doc): doc for doc in vector_results + bm25_results}
        sorted_docs = sorted(all_docs.items(),
                             key=lambda x: rrf_scores.get(x[0], 0),
                             reverse=True)

        return [doc for _, doc in sorted_docs]

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