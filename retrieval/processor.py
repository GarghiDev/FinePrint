from typing import List, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
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


class DocumentProcessor:
    """Process documents into chunks with embeddings and search indices."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = get_embedding_model()

    def process_document(
        self, doc_path: str
    ) -> Tuple[List[str], faiss.IndexFlatL2, BM25Okapi, np.ndarray]:
        """
        Process a document into chunks and create search indices.

        Args:
            doc_path: Path to the document file

        Returns:
            Tuple of (chunks, faiss_index, bm25_index, embeddings)
        """
        # Read and parse document
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Create chunks
        chunks = self._create_chunks(text)

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        # Create FAISS index
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)

        # Create BM25 index
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25_index = BM25Okapi(tokenized_chunks)

        return chunks, faiss_index, bm25_index, embeddings

    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Get chunk
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary if possible
            if end < text_length:
                # Look for sentence endings in the last 100 characters
                last_period = chunk.rfind(". ")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size * 0.7:  # Only break if not too early
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

        return chunks
