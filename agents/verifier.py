import os
from typing import List, Dict, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class VerificationAgent:
    """Verify research agent's answers against source chunks to detect hallucinations."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        # Using gemini-2.5-flash for free tier compatibility
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def verify_answer(
        self, query: str, answer: str, sources: List[Dict], document_name: str
    ) -> Dict[str, any]:
        """
        Verify answer against source chunks to detect contradictions or hallucinations.

        Args:
            query: Original user question
            answer: Generated answer from research agent
            sources: Retrieved source chunks
            document_name: Name of the privacy policy

        Returns:
            Dict with 'verified', 'issues', 'feedback', and 'confidence'
        """
        # Format sources
        context = self._format_sources(sources)

        # Create verification prompt
        prompt = f"""You are a fact-checking agent. Your job is to verify whether an answer is fully supported by the provided source text.

ORIGINAL QUESTION: {query}

ANSWER TO VERIFY:
{answer}

SOURCE TEXT FROM {document_name.upper()}:
{context}

VERIFICATION TASK:
Carefully check each claim in the answer against the source text. Identify:
1. Claims that are NOT supported by the source text
2. Claims that contradict the source text
3. Claims that add information not present in the source
4. Misquoted or misrepresented information

Respond in the following format:

VERIFICATION STATUS: [VERIFIED or CONTRADICTIONS_FOUND]

ISSUES FOUND:
[List each specific issue, or write "None" if verified]

DETAILED FEEDBACK:
[Explain what needs to be corrected, or write "Answer is accurate and fully supported by sources"]

CONFIDENCE: [0-100]%"""

        try:
            response = self.model.generate_content(prompt)
            verification_text = response.text

            # Parse verification result
            result = self._parse_verification(verification_text)
            result["full_verification"] = verification_text

            return result

        except Exception as e:
            return {
                "verified": False,
                "issues": [f"Verification error: {str(e)}"],
                "feedback": "Unable to verify due to error",
                "confidence": 0,
                "error": True,
            }

    def _format_sources(self, sources: List[Dict]) -> str:
        """Format source chunks for verification."""
        source_parts = []

        for chunk_data in sources:
            chunk_text = chunk_data["chunk"]
            chunk_idx = chunk_data["index"]

            source_parts.append(f"[Source Chunk #{chunk_idx}]\n{chunk_text}\n")

        return "\n".join(source_parts)

    def _parse_verification(self, verification_text: str) -> Dict[str, any]:
        """Parse verification response from Gemini."""
        lines = verification_text.split("\n")

        verified = False
        issues = []
        feedback = ""
        confidence = 50

        current_section = None

        for line in lines:
            line_stripped = line.strip()

            if "VERIFICATION STATUS:" in line_stripped:
                if (
                    "VERIFIED" in line_stripped
                    and "CONTRADICTIONS_FOUND" not in line_stripped
                ):
                    verified = True
                current_section = "status"

            elif "ISSUES FOUND:" in line_stripped:
                current_section = "issues"

            elif "DETAILED FEEDBACK:" in line_stripped:
                current_section = "feedback"

            elif "CONFIDENCE:" in line_stripped:
                current_section = "confidence"
                # Extract percentage
                try:
                    conf_str = line_stripped.split("CONFIDENCE:")[1].strip()
                    confidence = int(conf_str.replace("%", "").strip())
                except:
                    confidence = 50

            elif current_section == "issues" and line_stripped:
                if line_stripped.lower() != "none":
                    issues.append(line_stripped)

            elif current_section == "feedback" and line_stripped:
                if not line_stripped.startswith("CONFIDENCE:"):
                    feedback += line_stripped + " "

        # If issues found, mark as not verified
        if issues and issues[0].lower() != "none":
            verified = False

        return {
            "verified": verified,
            "issues": issues if issues else [],
            "feedback": feedback.strip(),
            "confidence": confidence,
        }

    def quick_check(self, answer: str, sources: List[Dict]) -> bool:
        """
        Quick sanity check - ensure answer isn't completely disconnected from sources.

        Returns:
            True if answer appears to reference sources, False otherwise
        """
        if not answer or not sources:
            return False

        # Check if answer contains any significant phrases from sources
        answer_lower = answer.lower()

        for source in sources:
            chunk_text = source["chunk"].lower()

            # Extract key phrases (simplified)
            words = chunk_text.split()
            for i in range(len(words) - 2):
                phrase = " ".join(words[i : i + 3])
                if phrase in answer_lower:
                    return True

        return False
