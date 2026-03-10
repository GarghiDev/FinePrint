import os
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class ResearchAgent:
    """Generate answers to user queries based on retrieved privacy policy chunks."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        # Using gemini-2.5-flash for free tier compatibility
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_answer(
        self, query: str, retrieved_chunks: List[Dict], document_name: str
    ) -> Dict[str, any]:
        """
        Generate an answer based on retrieved chunks.

        Args:
            query: User's question
            retrieved_chunks: List of relevant chunks from hybrid retrieval
            document_name: Name of the privacy policy document

        Returns:
            Dict with 'answer', 'citations', and 'sources'
        """
        # Format context from retrieved chunks
        context = self._format_context(retrieved_chunks)

        # Create prompt
        prompt = f"""You are a privacy policy analyst. Answer the user's question based ONLY on the provided context from the {document_name} privacy policy.

IMPORTANT INSTRUCTIONS:
1. Only use information explicitly stated in the context below
2. Cite specific passages by including the chunk index in brackets [#]
3. If the context doesn't contain enough information to answer, say so clearly
4. Be precise and specific - quote relevant phrases when appropriate
5. Do not make assumptions or add information not in the context

CONTEXT FROM {document_name.upper()}:
{context}

USER QUESTION: {query}

ANSWER (with citations):"""

        try:
            # Generate response
            response = self.model.generate_content(prompt)
            answer_text = response.text

            # Extract citations (chunk indices mentioned in the answer)
            citations = self._extract_citations(answer_text, retrieved_chunks)

            result = {
                "answer": answer_text,
                "citations": citations,
                "sources": retrieved_chunks,
                "document": document_name,
            }

            return result

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "citations": [],
                "sources": [],
                "document": document_name,
                "error": True,
            }

    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string with indices."""
        context_parts = []

        for i, chunk_data in enumerate(chunks):
            chunk_text = chunk_data["chunk"]
            chunk_idx = chunk_data["index"]
            score = chunk_data["score"]

            context_parts.append(
                f"[Chunk #{chunk_idx}] (Relevance: {score:.3f})\n{chunk_text}\n"
            )

        return "\n".join(context_parts)

    def _extract_citations(self, answer: str, chunks: List[Dict]) -> List[int]:
        """Extract chunk indices mentioned in the answer."""
        citations = []

        for chunk_data in chunks:
            chunk_idx = chunk_data["index"]
            # Look for references like [#123] or [Chunk #123]
            if f"#{chunk_idx}]" in answer or f"[{chunk_idx}]" in answer:
                citations.append(chunk_idx)

        return list(set(citations))  # Remove duplicates

    def refine_answer(
        self,
        original_query: str,
        previous_answer: str,
        verification_feedback: str,
        retrieved_chunks: List[Dict],
        document_name: str,
    ) -> Dict[str, any]:
        """
        Refine answer based on verification feedback.

        Args:
            original_query: User's original question
            previous_answer: Previously generated answer
            verification_feedback: Feedback from verification agent
            retrieved_chunks: Original retrieved chunks
            document_name: Name of the document

        Returns:
            Refined answer dict
        """
        context = self._format_context(retrieved_chunks)

        prompt = f"""You are a privacy policy analyst. Your previous answer had some issues identified by verification.

ORIGINAL QUESTION: {original_query}

YOUR PREVIOUS ANSWER:
{previous_answer}

VERIFICATION FEEDBACK (Issues Found):
{verification_feedback}

CONTEXT FROM {document_name.upper()}:
{context}

Please provide a CORRECTED answer that:
1. Addresses the verification feedback
2. Only uses information explicitly in the context
3. Includes proper citations [#]
4. Is accurate and verifiable

CORRECTED ANSWER:"""

        try:
            response = self.model.generate_content(prompt)
            answer_text = response.text
            citations = self._extract_citations(answer_text, retrieved_chunks)

            return {
                "answer": answer_text,
                "citations": citations,
                "sources": retrieved_chunks,
                "document": document_name,
                "refined": True,
            }

        except Exception as e:
            return {
                "answer": f"Error refining answer: {str(e)}",
                "citations": [],
                "sources": [],
                "document": document_name,
                "error": True,
            }
