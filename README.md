# FinePrint - Multi-Agent Privacy Policy Analyzer

A sophisticated RAG (Retrieval-Augmented Generation) system that reads privacy policies from TikTok, Meta, and Spotify, and answers user questions with verified, hallucination-checked responses.

## Features

- **Hybrid Retrieval**: Combines BM25 keyword search and FAISS vector search for optimal accuracy
- **Multi-Agent Architecture**: Separate Research and Verification agents with self-correction loop
- **Hallucination Detection**: Verification agent cross-checks every answer against source material
- **Self-Correction**: Automatically refines answers up to 2 times if verification fails
- **Interactive UI**: Clean Streamlit interface with document selector and chat
- **Source Citations**: Every answer includes citations and source chunks

## Architecture

```
User Query → Hybrid Retriever (BM25 + FAISS) → Research Agent (Gemini) 
          → Verification Agent → [If issues found] → Refine Answer → Verify Again
          → [Max 2 retries] → Final Answer with Verification Status
```

## Tech Stack

- **LLM**: Google Gemini 1.5 Pro (via google-generativeai)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Keyword Search**: BM25 (rank_bm25)
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Orchestration**: LangGraph with state management
- **UI**: Streamlit
- **Document Parsing**: Docling

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in `.env`:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

1. Wait for the privacy policies to load (happens automatically on first run)
2. Select a privacy policy from the sidebar (TikTok, Meta, or Spotify)
3. Ask questions in the chat interface
4. View answers with verification status, confidence scores, and source citations

## Project Structure

```
fineprint/
├── app.py                  # Streamlit UI
├── agents/
│   ├── __init__.py
│   ├── research.py         # Research Agent (generates answers)
│   ├── verifier.py         # Verification Agent (checks for hallucinations)
│   └── workflow.py         # LangGraph orchestration
├── retrieval/
│   ├── __init__.py
│   ├── processor.py        # Document parsing, chunking, embedding
│   └── hybrid.py           # BM25 + FAISS hybrid retriever
├── data/
│   ├── TikTok_Policy.txt
│   ├── Meta_Policy.txt
│   └── Spotify_Policy.txt
├── requirements.txt
├── .env
└── README.md
```

## How It Works

### 1. Document Processing
- Privacy policies are parsed using Docling
- Text is chunked into 500-word segments with 50-word overlap
- Chunks are embedded using sentence-transformers
- FAISS index and BM25 index are built for hybrid search

### 2. Hybrid Retrieval
- User query is processed by both BM25 (keyword) and FAISS (semantic) search
- Results are merged with weighted scoring (default: 50/50)
- Top 5 most relevant chunks are returned

### 3. Research Agent
- Gemini receives the query and retrieved chunks
- Generates an answer with citations [#chunk_id]
- Only uses information explicitly in the provided context

### 4. Verification Agent
- Cross-checks every claim in the answer against source chunks
- Identifies contradictions, hallucinations, or unsupported claims
- Returns verification status, issues found, and confidence score

### 5. Self-Correction Loop
- If verification fails, feedback is sent back to Research Agent
- Research Agent refines the answer (max 2 retries)
- Process repeats until verified or max retries reached

### 6. UI Display
- Shows answer with verification badge (✓ VERIFIED or ⚠ UNVERIFIED)
- Displays confidence score (0-100%)
- Lists any verification issues found
- Shows source chunks with citations highlighted

## Example Queries

- "How does TikTok use my personal data?"
- "What information does Meta collect about me?"
- "Can I delete my data from Spotify?"
- "How long does TikTok retain my information?"
- "Does Meta share my data with third parties?"

## License

MIT
