from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from agents.research import ResearchAgent
from agents.verifier import VerificationAgent
from retrieval.hybrid import HybridRetriever


class WorkflowState(TypedDict):
    """State for the LangGraph workflow."""

    query: str
    document_name: str
    retrieved_chunks: List[Dict]
    answer: str
    citations: List[int]
    sources: List[Dict]
    verified: bool
    verification_feedback: str
    verification_issues: List[str]
    confidence: int
    retry_count: int
    max_retries: int
    error: str
    final_result: Dict


class PrivacyPolicyWorkflow:
    """LangGraph workflow orchestrating retrieval, research, and verification."""

    def __init__(
        self, retriever: HybridRetriever, document_name: str, max_retries: int = 2
    ):
        self.retriever = retriever
        self.document_name = document_name
        self.max_retries = max_retries
        self.research_agent = ResearchAgent()
        self.verification_agent = VerificationAgent()

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("verify", self.verify_node)
        workflow.add_node("refine", self.refine_node)
        workflow.add_node("finalize", self.finalize_node)

        # Set entry point
        workflow.set_entry_point("retrieve")

        # Add edges
        workflow.add_edge("retrieve", "research")
        workflow.add_edge("research", "verify")

        # Conditional edges from verify
        workflow.add_conditional_edges(
            "verify", self.should_refine, {"refine": "refine", "finalize": "finalize"}
        )

        workflow.add_edge("refine", "verify")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def retrieve_node(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant chunks using hybrid search."""
        try:
            retrieved_chunks = self.retriever.hybrid_search(
                query=state["query"], top_k=5
            )

            state["retrieved_chunks"] = retrieved_chunks
            state["sources"] = retrieved_chunks

        except Exception as e:
            state["error"] = f"Retrieval error: {str(e)}"
            state["retrieved_chunks"] = []

        return state

    def research_node(self, state: WorkflowState) -> WorkflowState:
        """Generate answer using Research Agent."""
        try:
            result = self.research_agent.generate_answer(
                query=state["query"],
                retrieved_chunks=state["retrieved_chunks"],
                document_name=state["document_name"],
            )

            state["answer"] = result["answer"]
            state["citations"] = result["citations"]
            state["sources"] = result["sources"]

            if result.get("error"):
                state["error"] = result["answer"]

        except Exception as e:
            state["error"] = f"Research error: {str(e)}"
            state["answer"] = ""

        return state

    def verify_node(self, state: WorkflowState) -> WorkflowState:
        """Verify answer using Verification Agent."""
        try:
            verification = self.verification_agent.verify_answer(
                query=state["query"],
                answer=state["answer"],
                sources=state["sources"],
                document_name=state["document_name"],
            )

            state["verified"] = verification["verified"]
            state["verification_feedback"] = verification["feedback"]
            state["verification_issues"] = verification["issues"]
            state["confidence"] = verification["confidence"]

        except Exception as e:
            state["error"] = f"Verification error: {str(e)}"
            state["verified"] = False
            state["confidence"] = 0

        return state

    def refine_node(self, state: WorkflowState) -> WorkflowState:
        """Refine answer based on verification feedback."""
        try:
            refined_result = self.research_agent.refine_answer(
                original_query=state["query"],
                previous_answer=state["answer"],
                verification_feedback=state["verification_feedback"],
                retrieved_chunks=state["retrieved_chunks"],
                document_name=state["document_name"],
            )

            state["answer"] = refined_result["answer"]
            state["citations"] = refined_result["citations"]
            state["retry_count"] += 1

        except Exception as e:
            state["error"] = f"Refinement error: {str(e)}"

        return state

    def finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Prepare final result."""
        state["final_result"] = {
            "query": state["query"],
            "answer": state["answer"],
            "citations": state["citations"],
            "sources": state["sources"],
            "verified": state["verified"],
            "confidence": state["confidence"],
            "verification_feedback": state["verification_feedback"],
            "verification_issues": state["verification_issues"],
            "retry_count": state["retry_count"],
            "document": state["document_name"],
        }

        return state

    def should_refine(self, state: WorkflowState) -> str:
        """Decide whether to refine the answer or finalize."""
        # If already verified, finalize
        if state["verified"]:
            return "finalize"

        # If max retries reached, finalize with current answer
        if state["retry_count"] >= state["max_retries"]:
            return "finalize"

        # If there's a critical error, finalize
        if state.get("error"):
            return "finalize"

        # Otherwise, refine
        return "refine"

    def run(self, query: str) -> Dict:
        """
        Run the complete workflow.

        Args:
            query: User's question about the privacy policy

        Returns:
            Final result dictionary with answer, verification status, etc.
        """
        # Initialize state
        initial_state = {
            "query": query,
            "document_name": self.document_name,
            "retrieved_chunks": [],
            "answer": "",
            "citations": [],
            "sources": [],
            "verified": False,
            "verification_feedback": "",
            "verification_issues": [],
            "confidence": 0,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "error": "",
            "final_result": {},
        }

        # Run workflow
        final_state = self.workflow.invoke(initial_state)

        return final_state["final_result"]
