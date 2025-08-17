import os
from pathlib import Path
import json
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    pipeline, # Keep pipeline import for now, but we won't use it directly in generate_answer
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
import torch

import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)


import hashlib  # For custom hashing


# ---------- Environment & Config ----------
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    print("Hugging Face token loaded successfully!")
    #print(f"Token: {hf_token}")
else:
    print("Failed to load Hugging Face token. Check your .env file.")

st.set_page_config(page_title="Katy ISD Chatbot", page_icon="🎓", layout="wide")
st.image("jhs.png", width=120)
st.markdown(
    "<h1 style='text-align: center;'>Katy ISD Website Chatbot 🎓🤖</h1>",
    unsafe_allow_html=True,
)
st.write("Ask anything about the Katy ISD website and get instant answers!")


# ---------- LLM Loader ----------
@st.cache_resource
def load_llm():
    model_name = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", token=hf_token
    )
    return tokenizer, model


# ---------- Document Loader ----------
@st.cache_resource
def load_documents():
    loaders = []
    all_loaded_docs = []  # Use a single list to collect all documents

    # Load PDFs from 'documents/' subfolder within 'website_content'
    pdf_dir = (
        Path("website_content") / "documents"
    )  # Use Path for cleaner path handling
    if pdf_dir.is_dir():  # Check if directory exists
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                loaders.append(
                    PyPDFLoader(str(pdf_dir / filename))
                )  # Convert Path to string for PyPDFLoader

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
            # st.info(f"Loaded {len(loaded_docs)} documents from {loader.__class__.__name__} for {loader.file_path if hasattr(loader, 'file_path') else 'unknown source'}")
        except Exception as e:
            st.error(
                f"Error loading document with {loader.__class__.__name__} from {loader.file_path if hasattr(loader, 'file_path') else 'unknown source'}: {e}"
            )

    if not all_loaded_docs:
        st.warning(
            "No documents were loaded. Please check the 'website_content' directory and file permissions/formats."
        )

    return all_loaded_docs


# --- Custom Hashing Function for LangChain Document objects ---
def hash_langchain_document(doc):
    content_hash = hashlib.sha256(
        doc.page_content.encode("utf-8", errors="ignore")
    ).hexdigest()

    metadata_string = str(sorted(doc.metadata.items()))
    metadata_hash = hashlib.sha256(
        metadata_string.encode("utf-8", errors="ignore")
    ).hexdigest()

    return f"doc-{content_hash}-{metadata_hash}"


# ---------- FAISS Vector Store with disk caching ----------
@st.cache_resource(hash_funcs={Document: hash_langchain_document})
def create_vector_store(documents=None):
    index_path = Path("faiss_index")
    metadata_file = index_path / "metadata.json"

    embedder_model = "BAAI/bge-base-en-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedder_model, model_kwargs={"device": device}
    )

    # --- Step 1: Check if index exists and is compatible ---
    if index_path.exists():
        try:
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                previous_model = metadata.get("embedder", "")
                if previous_model != embedder_model:
                    st.warning(f"Embedder changed from '{previous_model}' to '{embedder_model}'. Rebuilding FAISS index.")
                    shutil.rmtree(index_path)
                    raise FileNotFoundError
            else:
                st.warning("No metadata found for FAISS index. Rebuilding index.")
                shutil.rmtree(index_path)
                raise FileNotFoundError

            with st.spinner("Loading FAISS vector store from disk..."):
                vectorstore = FAISS.load_local(
                    str(index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                st.success("FAISS index loaded from disk.")
                return vectorstore

        except FileNotFoundError:
            pass  # Will fall through to index creation below

    # --- Step 2: Load documents if not already provided ---
    if documents is None:
        documents = load_documents()
        if not documents:
            st.error("No documents found to create FAISS index.")
            return None

    # --- Step 3: Create new index ---
    with st.spinner("Creating new FAISS index..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        if not chunks:
            st.error("No chunks generated from documents.")
            return None

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(index_path))

        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump({"embedder": embedder_model}, f)

        st.success("FAISS index created and saved.")

    return vectorstore




# ---------- Answer Generator ----------
def generate_answer(query, retriever, tokenizer, model):
    response = "I apologize, but I couldn't process that request at the moment. Please try again."
    docs = []

    # Step 1: Retrieve relevant documents for the query
    with st.spinner("Searching for relevant information..."):
        try:
            docs = retriever.similarity_search(query, k=5)
            # You can add a threshold here if needed, e.g., if max_score < some_value
        except Exception as e:
            st.error(f"Error during document retrieval: {e}")
            docs = []

    # Step 2: Clean and build context from retrieved documents
    context = ""
    retrieved_sources = []
    if docs: # Only build context if documents were actually retrieved
        cleaned_contents = []
        for doc in docs:
            cleaned_content = doc.page_content.strip()
            if cleaned_content.startswith("Content:"):
                cleaned_content = cleaned_content[len("Content:") :].strip()
            if cleaned_content:
                cleaned_contents.append(cleaned_content)
                retrieved_sources.append(doc.metadata.get("source", "Unknown Source"))
        context = "\n\n---\n\n".join(cleaned_contents)
    
    st.write("### Retrieved Chunks")
    for i, doc in enumerate(docs):
        st.markdown(f"**Doc {i+1} Source:** {doc.metadata.get('source', 'Unknown')}")
        st.text(doc.page_content[:600])

    # Step 3: Construct a chat-style prompt suitable for Gemma-7B-It
    messages = []
    if context.strip():
        # If context is available, use the RAG prompt
        messages.append(
            {"role": "user", "content": f"You are an AI assistant specialized in providing information about Katy ISD. Your task is to answer the user's question concisely, using only the context provided. If the answer cannot be found in the context, clearly state that you do not have enough information to answer. Do not make up information.\n\nContext:\n---\n{context}\n---\n\nQuestion: {query}"}
        )
    else:
        # If no context is found, let the model try to answer without context
        # or indicate it can't find specific information.
        # This allows for conversational greetings or general knowledge answers if the model has them.
        messages.append(
            {"role": "user", "content": f"You are an AI assistant specialized in providing information about Katy ISD. I do not have specific information from documents to answer your question. If your question is a greeting or general inquiry, please respond politely. If it's a specific question about Katy ISD that requires factual data, state that you cannot find relevant information in your knowledge base. \n\nQuestion: {query}"}
        )


    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Step 4: Generate answer using manual tokenizer and model.generate
    try:
        with st.spinner("Generating answer..."):
            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

            output_ids = model.generate(
                **input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                # Set specific stop tokens for Gemma to prevent it from generating another turn
                # "<end_of_turn>" or "<start_of_turn>"
                eos_token_id=tokenizer.eos_token_id, # Ensure generation stops at EOS token
                # This might be needed if Gemma doesn't naturally stop after its turn:
                # stopping_criteria=StoppingCriteriaList([StopOnTokens(tokenizer, ["<end_of_turn>"])]),
            )

            full_generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

            model_turn_start_tag = "<start_of_turn>model\n"
            model_turn_start_index = full_generated_text.find(model_turn_start_tag)

            if model_turn_start_index != -1:
                raw_response = full_generated_text[model_turn_start_index + len(model_turn_start_tag):].strip()

                # Clean by removing potential <end_of_turn> or other turns
                if "<end_of_turn>" in raw_response:
                    raw_response = raw_response.split("<end_of_turn>")[0].strip()
                if "<start_of_turn>" in raw_response:
                    raw_response = raw_response.split("<start_of_turn>")[0].strip()
                
                # Finally, remove any remaining special tokens if skip_special_tokens=False was not enough
                response = tokenizer.decode(tokenizer.encode(raw_response), skip_special_tokens=True).strip()

            else:
                st.warning("Could not find the start of the model's response in the generated text.")
                response = "I couldn't generate a clear response from the model. Please try again."

    except Exception as e:
        st.error(f"Error generating LLM response: {e}")
        response = "I'm sorry, an error occurred while generating the answer. Please try rephrasing your question or check the server status."

    return response, docs


# ---------- Main App Flow ----------
tokenizer_llm, model_llm = load_llm()
#documents = load_documents()

#docs_to_process = documents

vectorstore = create_vector_store() 

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("question_form", clear_on_submit=True):
    user_query = st.text_input("Ask a question about Katy ISD...", key="query_input")
    submitted = st.form_submit_button("Submit")

if submitted and user_query:
    if vectorstore is not None:
        answer, sources = generate_answer(user_query, vectorstore, tokenizer_llm, model_llm)
        st.session_state.chat_history.append((user_query, answer, sources))
    else:
        st.warning("Cannot answer: Vector store not initialized. Please ensure documents are loaded and processed correctly.")


# ---------- Display Chat ----------
for user, bot, srcs in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)
        # with st.expander("Sources"): # Uncomment if you want to display sources
        #     if srcs:
        #         for doc in srcs:
        #             st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
        #     else:
        #         st.info("No specific sources were retrieved for this answer.")