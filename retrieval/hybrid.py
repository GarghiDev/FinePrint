from typing import List, Dict, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import streamlit as st
import os

# Disable GPU usage for PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@st.cache_resource
def get_embedding_model():
    """Load and cache the embedding model."""
    import torch

    # Force CPU usage
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # Load model without device parameter first
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Then explicitly move to CPU
    model = model.to("cpu")
    return model


class HybridRetriever:
    """Combine BM25 keyword search and FAISS vector search for hybrid retrieval."""

    def __init__(
        self,
        chunks: List[str],
        faiss_index: faiss.IndexFlatL2,
        bm25_index: BM25Okapi,
        embeddings: np.ndarray,
    ):
        self.chunks = chunks
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.embeddings = embeddings
        self.embedding_model = get_embedding_model()

    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Perform BM25 keyword search."""
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices and scores
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]

        return results

    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Perform FAISS vector similarity search."""
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.faiss_index.search(query_embedding, top_k)

        # Convert distances to similarity scores (inverse of L2 distance)
        results = [
            (int(indices[0][i]), 1.0 / (1.0 + float(distances[0][i])))
            for i in range(len(indices[0]))
        ]

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> List[Dict]:
        """
        Combine BM25 and vector search results with weighted scoring.

        Args:
            query: Search query
            top_k: Number of results to return
            bm25_weight: Weight for BM25 scores (default 0.5)
            vector_weight: Weight for vector scores (default 0.5)

        Returns:
            List of dicts with chunk text, score, and index
        """
        # Get results from both methods
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        vector_results = self.vector_search(query, top_k=top_k * 2)

        # Normalize scores
        bm25_scores = self._normalize_scores([score for _, score in bm25_results])
        vector_scores = self._normalize_scores([score for _, score in vector_results])

        # Create score dictionaries
        bm25_dict = {idx: score for (idx, _), score in zip(bm25_results, bm25_scores)}
        vector_dict = {
            idx: score for (idx, _), score in zip(vector_results, vector_scores)
        }

        # Combine scores
        all_indices = set(bm25_dict.keys()) | set(vector_dict.keys())
        combined_scores = {}

        for idx in all_indices:
            bm25_score = bm25_dict.get(idx, 0.0)
            vector_score = vector_dict.get(idx, 0.0)
            combined_scores[idx] = (bm25_weight * bm25_score) + (
                vector_weight * vector_score
            )

        # Sort by combined score and get top-k
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        # Format results
        final_results = []
        for idx, score in sorted_results:
            final_results.append(
                {"chunk": self.chunks[idx], "score": score, "index": idx}
            )

        return final_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not scores or max(scores) == min(scores):
            return [0.0] * len(scores)

        min_score = min(scores)
        max_score = max(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def get_context_window(
        self, chunk_indices: List[int], window_size: int = 1
    ) -> List[str]:
        """Get surrounding chunks for context."""
        expanded_chunks = []

        for idx in chunk_indices:
            start = max(0, idx - window_size)
            end = min(len(self.chunks), idx + window_size + 1)

            for i in range(start, end):
                if self.chunks[i] not in expanded_chunks:
                    expanded_chunks.append(self.chunks[i])

        return expanded_chunks
