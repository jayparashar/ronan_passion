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
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import hashlib
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ---------- Load Environment Variables ----------
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# ---------- Streamlit UI Config ----------
st.set_page_config(page_title="Katy ISD Chatbot", page_icon="🎓", layout="wide")
st.image("katyisd.jpg", width=120)
st.markdown("<h1 style='text-align: center;'>Katy ISD Website Chatbot 🎓🤖</h1>", unsafe_allow_html=True)
st.write("Ask anything about the Katy ISD website and get instant answers!")

# ---------- Load Mistral LLM ----------
@st.cache_resource
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    return tokenizer, model

# ---------- Load Documents ----------
@st.cache_resource
def load_documents():
    loaders, all_docs = [], []
    pdf_dir = Path("website_content/documents")
    root_dir = Path("website_content")

    if pdf_dir.exists():
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                loaders.append(PyPDFLoader(str(pdf_dir / filename)))

    if root_dir.exists():
        for filename in os.listdir(root_dir):
            path = root_dir / filename
            if path.suffix.lower() == ".txt":
                loaders.append(TextLoader(str(path), encoding="utf-8"))
            elif path.suffix.lower() == ".docx":
                loaders.append(Docx2txtLoader(str(path)))

    for loader in loaders:
        try:
            all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {loader.__class__.__name__}: {e}")
    return all_docs

# ---------- Document Hashing ----------
def hash_langchain_document(doc):
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    metadata_hash = hashlib.sha256(str(sorted(doc.metadata.items())).encode("utf-8")).hexdigest()
    return f"doc-{content_hash}-{metadata_hash}"

# ---------- FAISS Vector Store ----------
@st.cache_resource(hash_funcs={Document: hash_langchain_document})
def create_vector_store(documents=None):
    index_path = Path("faiss_index")
    metadata_file = index_path / "metadata.json"

    embedder_model = "BAAI/bge-base-en-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name=embedder_model, model_kwargs={"device": device})

    if index_path.exists() and metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
        if meta.get("embedder") == embedder_model:
            return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        shutil.rmtree(index_path)

    if documents is None:
        documents = load_documents()
        if not documents:
            st.error("No documents found.")
            return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_path))
    with open(metadata_file, "w") as f:
        json.dump({"embedder": embedder_model}, f)
    return vectorstore

# ---------- Generate Answer ----------
def generate_answer(query, retriever, tokenizer, model):
    docs = retriever.similarity_search(query, k=10)
    context = "\n\n---\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())

    # st.write("### Retrieved Chunks")
    # for i, doc in enumerate(docs):
    #     st.markdown(f"**Doc {i+1} Source:** {doc.metadata.get('source', 'Unknown')}")
    #     st.text(doc.page_content[:600])

    # Mistral-style prompt
    prompt = f"""[INST] You are an assistant specialized in answering questions about Katy ISD based only on the provided context.
If the answer is not in the context, say "I do not have enough information to answer that question."
Do not make up information.

Context:
{context}

Question:
{query}
[/INST]"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if '[/INST]' in decoded:
            answer = decoded.split('[/INST]', 1)[1].strip()
        else:
            answer = decoded.strip()

        return answer, docs
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm sorry, I couldn't generate a response.", docs

# ---------- Main UI ----------
tokenizer_llm, model_llm = load_llm()
vectorstore = create_vector_store()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("question_form", clear_on_submit=True):
    user_query = st.text_input("Ask a question about Katy ISD...", key="query_input")
    submitted = st.form_submit_button("Submit")

if submitted and user_query:
    if vectorstore:
        answer, sources = generate_answer(user_query, vectorstore, tokenizer_llm, model_llm)
        st.session_state.chat_history.append((user_query, answer, sources))

# ---------- Chat Display ----------
for user, bot, srcs in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)
