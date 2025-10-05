# Katy ISD Chatbot — High School Passion Project

A Streamlit app that answers questions about the Katy ISD website using Retrieval-Augmented Generation (RAG). It ingests local documents, builds a FAISS index with bge embeddings, retrieves the most relevant chunks, and generates grounded answers with Mistral-7B-Instruct.

- Author: Your Name (High School Student)
- Tech: Streamlit, LangChain, FAISS, Hugging Face Transformers, PyTorch
- Models: BAAI/bge-base-en-v1.5 (embeddings), mistralai/Mistral-7B-Instruct-v0.1 (LLM)

## Overview
The app focuses on reliability: it answers only from the retrieved context. If the information isn’t in your local documents, it clearly says it doesn’t have enough information rather than guessing.

## Quickstart
Requirements:
- Python 3.10+
- .env with a valid Hugging Face token (HUGGINGFACE_TOKEN)
- GPU optional (CPU fallback supported, just slower)

Setup:
1) Install dependencies
   pip install -r requirements.txt

2) Create a .env file in the project root
   HUGGINGFACE_TOKEN=your_hf_token_here

3) Prepare your documents (scraped from the katyisd domain or your own set)
- PDFs → website_content/documents/
- TXT and DOCX → website_content/

4) Optional: Add a sidebar image
- Place katyisd.jpg in the project root

5) Run the app
   streamlit run app.py

On the first run, the app builds a FAISS index from your documents. Subsequent runs reuse the saved index.

## Project structure
- app.py
- website_content/
  - documents/           (PDF files)
  - any_txt_or_docx_here (TXT/DOCX files)
- faiss_index/           (auto-created and persisted)
- katyisd.jpg            (optional sidebar image)
- .env
- requirements.txt

## How it works (RAG flow)
1) Ingest: Load PDFs, TXTs, and DOCX from website_content/
2) Chunk: Split into ~1000 character chunks with 150 overlap
3) Embed + Index: Build a FAISS vector store with BAAI/bge-base-en-v1.5
4) Retrieve: Similarity search for the top 10 chunks per query
5) Generate: Mistral-7B-Instruct answers using only the retrieved context

## Features
- Multi-format ingestion: PDFs, TXT, DOCX
- Reliable retrieval: FAISS + bge-base-en-v1.5
- Guarded responses: answers only from provided context
- Caching: st.cache_resource for LLM, documents, and index
- Deterministic rebuild: index auto-rebuilds if the embedding model changes
- Chat UI: user/assistant bubbles, Clear Chat button, newest messages shown first

Note on sources: The app internally tracks the retrieved documents for each answer. You can optionally render the top sources under each response (see snippet below).

## Configuration (in code)
- Embeddings: BAAI/bge-base-en-v1.5
- Chunking: chunk_size=1000, chunk_overlap=150
- Retrieval k: 10
- LLM: mistralai/Mistral-7B-Instruct-v0.1
- Generation: max_new_tokens=512, temperature=0.7
- Guardrail prompt: If info isn’t in context, reply that there isn’t enough information

## Usage tips
- Place your source files in website_content/ and website_content/documents/
- Use the Clear Chat button to reset the conversation
- If you change documents significantly, delete faiss_index/ to force a rebuild

## Evaluation and testing
- Functional:
  - Ask a few known-answer questions and confirm correctness
  - Try one out-of-scope question; the app should say it lacks sufficient information
- Robustness:
  - Empty folders or a bad PDF should show a helpful error (no crash)
  - GPU absent: CPU fallback still works (slower)
- Performance:
  - First (cold) run is slower due to downloads and indexing
  - Subsequent (warm) runs are faster using cached resources

## Troubleshooting
- Hugging Face auth error:
  - Ensure .env contains HUGGINGFACE_TOKEN and the project has read access
- “No documents found.”:
  - Ensure website_content/ exists and contains PDFs/TXTs/DOCX
- Index mismatch:
  - If you change the embedding model, the app rebuilds automatically
  - To force rebuild, delete faiss_index/

## Security and privacy
- Token is loaded from .env (not hardcoded)
- No user authentication or PII storage
- Index is built from local files only

## Limitations
- Source citations are not displayed by default in the UI (optional snippet below)
- Large models may be slow on CPU
- No authentication (not for public internet deployment as-is)

## Roadmap
- Show source filenames/snippets below each answer
- Smaller CPU-friendly LLM fallback
- Add evaluation set and accuracy metrics
- Optional basic authentication

## Acknowledgments
- Streamlit, LangChain, FAISS, Hugging Face, Mistral-7B-Instruct, BAAI bge-base-en


