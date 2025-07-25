import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# No longer needed since using LangChain's loaders, but keeping this comment for clarity
# from pdfminer.high_level import extract_text as extract_pdf_text
# from docx import Document as DocxDocument

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import hashlib # For custom hashing


# ---------- Environment & Config ----------
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

st.set_page_config(page_title="Katy ISD Chatbot", page_icon="🎓", layout="wide")
st.image("jhs.png", width=120)
st.markdown("<h1 style='text-align: center;'>Katy ISD Website Chatbot 🎓🤖</h1>", unsafe_allow_html=True)
st.write("Ask anything about the Katy ISD website and get instant answers!")


# ---------- LLM Loader ----------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base" # Good choice for better answers
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token)
    # device=0 for GPU if available on RunPod, otherwise device=-1 for CPU
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)


# ---------- Document Loader ----------
@st.cache_resource
def load_documents():
    loaders = []
    all_loaded_docs = [] # Use a single list to collect all documents

    # Load PDFs from 'documents/' subfolder within 'website_content'
    pdf_dir = Path("website_content") / "documents" # Use Path for cleaner path handling
    if pdf_dir.is_dir(): # Check if directory exists
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                loaders.append(PyPDFLoader(str(pdf_dir / filename))) # Convert Path to string for PyPDFLoader

    # Load TXT and DOCX from root 'website_content/'
    root_dir = Path("website_content")
    if root_dir.is_dir():
        for filename in os.listdir(root_dir):
            path = root_dir / filename
            if path.suffix.lower() == ".txt":
                loaders.append(TextLoader(str(path), encoding="utf-8"))
            elif path.suffix.lower() == ".docx":
                loaders.append(Docx2txtLoader(str(path)))

    # Process all loaders
    for loader in loaders:
        try:
            loaded_docs = loader.load()
            all_loaded_docs.extend(loaded_docs)
            st.info(f"Loaded {len(loaded_docs)} documents from {loader.__class__.__name__} for {loader.file_path if hasattr(loader, 'file_path') else 'unknown source'}")
        except Exception as e:
            st.error(f"Error loading document with {loader.__class__.__name__} from {loader.file_path if hasattr(loader, 'file_path') else 'unknown source'}: {e}")

    if not all_loaded_docs:
        st.warning("No documents were loaded. Please check the 'website_content' directory and file permissions/formats.")

    return all_loaded_docs


# --- Custom Hashing Function for LangChain Document objects ---
# Essential for @st.cache_resource when using LangChain Document objects
def hash_langchain_document(doc):
    # Ensure all parts used for hashing are strings and consistently ordered
    content_hash = hashlib.sha256(doc.page_content.encode('utf-8', errors='ignore')).hexdigest()

    # Sort metadata items for consistent hashing, convert to string
    # Assuming metadata values are simple (strings, numbers, etc.) and hashable
    # If metadata can contain complex unhashable objects, this needs to be more robust
    metadata_string = str(sorted(doc.metadata.items()))
    metadata_hash = hashlib.sha256(metadata_string.encode('utf-8', errors='ignore')).hexdigest()

    return f"doc-{content_hash}-{metadata_hash}"


# ---------- FAISS Vector Store with disk caching ----------
# @st.cache_resource is now correctly applied with hash_funcs
@st.cache_resource(hash_funcs={Document: hash_langchain_document})
def create_vector_store(documents):
    index_path = Path("faiss_index")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    if index_path.exists():
        with st.spinner("Loading FAISS vector store from disk..."):
            # Ensure allow_dangerous_deserialization is set to True if loading from untrusted sources
            # or if LangChain version might change often, leading to serialization issues.
            # Best practice: Only use if you trust your index.
            vectorstore = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True # Be cautious with this in production
            )
            st.success("FAISS index loaded from disk.")
    else:
        with st.spinner("Creating new FAISS index... This may take a few minutes for large datasets."):
            if not documents: # Handle case where no documents were loaded
                st.error("Cannot create vector store: No documents provided.")
                return None # Return None if no documents

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            if not chunks: # Handle case where splitting results in no chunks
                st.error("Cannot create vector store: No chunks generated from documents. Check chunk size/overlap or document content.")
                return None

            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(str(index_path))
            st.success("FAISS index created and saved.")

    return vectorstore


# ---------- Answer Generator (Rewritten for better prompt & error handling) ----------
def generate_answer(query, retriever, llm_pipeline):
    response = "I apologize, but I couldn't process that request at the moment. Please try again." # Default error message
    docs = [] # Default empty list for sources

    with st.spinner("Searching for relevant information..."):
        try:
            # Retrieve more documents (k=5 instead of 3) to give LLM more context options
            docs = retriever.similarity_search(query, k=5)
        except Exception as e:
            st.error(f"Error during document retrieval: {e}")
            # If retrieval fails, proceed with empty context
            docs = []

    st.write("### Debug: Retrieved Documents")
    if not docs:
        st.warning("No documents retrieved for the query.")
        context = "" # Set context to empty if no docs
        retrieved_sources = []
    else:
        cleaned_contents = []
        retrieved_sources = []
        for i, doc in enumerate(docs):
            st.write(f"--- Document {i+1} ---")
            source_info = doc.metadata.get('source', 'Unknown Source')
            st.write(f"Source: {source_info}")
            st.write(f"Content length: {len(doc.page_content)}")

            # Basic cleaning: strip whitespace, remove common "Content:" prefix
            cleaned_content = doc.page_content.strip()
            # If the specific "Content:" prefix is still an issue, handle it.
            if cleaned_content.startswith("Content:"):
                cleaned_content = cleaned_content[len("Content:"):].strip()

            st.code(cleaned_content[:300] + "...") if cleaned_content else st.code("Content is empty or too short after cleaning.")
            st.write("---")

            if cleaned_content: # Only add non-empty content to context
                cleaned_contents.append(cleaned_content)
                retrieved_sources.append(source_info) # Collect sources for display

        # Join cleaned contents, ensuring distinct sources are listed
        context = "\n\n---\n\n".join(cleaned_contents) # Use a clearer delimiter for context sections

        # Ensure unique sources are displayed (though docs retains original for direct source display)
        # This retrieved_sources list isn't directly used for display, but could be.
        # retrieved_sources = list(set(retrieved_sources)) # Not strictly needed here, `docs` is used later

    st.write("### Debug: Context used:")
    st.code(context) # Display the full context being sent to the LLM

    # --- IMPROVED PROMPT TEMPLATE ---
    if context.strip():
        prompt = f"""You are an AI assistant specialized in providing information about Katy ISD.
Your task is to answer the user's question based *only* on the provided context.
If the answer cannot be found in the context, clearly state that you do not have enough information and briefly explain why.
Do not make up answers. Provide detailed and comprehensive responses when the context allows.

Context:
---
{context}
---

Question: {query}

Answer:"""
    else:
        # Fallback prompt when no relevant documents are found
        prompt = f"""You are an AI assistant specialized in providing information about Katy ISD.
No relevant information was found in the provided documents to answer the following question.
Please state that you cannot answer the question based on the available information.
Do not make up answers.

Question: {query}

Answer:"""


    st.write("### Debug: Prompt sent to LLM:")
    st.code(prompt) # Display the full prompt

    # Add a spinner for LLM generation with error handling
    try:
        with st.spinner("Generating answer..."):
            # Increase max_new_tokens for potentially longer answers
            # Adjust as needed, higher values use more resources
            llm_response_list = llm_pipeline(prompt, max_new_tokens=512)
            response = llm_response_list[0]['generated_text']
    except Exception as e:
        st.error(f"Error generating LLM response: {e}")
        response = "I'm sorry, an error occurred while generating the answer. Please try rephrasing your question or check the server status."

    st.write("### Debug: LLM Response:")
    st.code(response) # Display the raw LLM response

    # ALWAYS return a tuple (string, list) to avoid TypeError
    return response, docs


# ---------- Main App Flow ----------
llm_pipeline = load_llm()
documents = load_documents()

# Removed [:50] limit here to ensure all loaded documents are processed by default.
docs_to_process = documents

vectorstore = None # Initialize vectorstore to None
if not docs_to_process:
    st.error("No documents were loaded or prepared for the vector store. Please check 'website_content' directory and file formats.")
else:
    vectorstore = create_vector_store(docs_to_process)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask a question about Katy ISD...", key="input")

if user_query:
    if vectorstore is not None: # Only proceed if vectorstore was successfully created
        # The key change is here: generate_answer is now guaranteed to return a tuple
        answer, sources = generate_answer(user_query, vectorstore, llm_pipeline)
        st.session_state.chat_history.append((user_query, answer, sources))
    else:
        st.warning("Cannot answer: Vector store not initialized. Please ensure documents are loaded and processed correctly.")

# ---------- Display Chat ----------
for user, bot, srcs in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)
        with st.expander("Sources"):
            if srcs: # Ensure sources list is not empty
                for doc in srcs:
                    st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
            else:
                st.info("No specific sources were retrieved for this answer.")