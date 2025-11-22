# streamlit_rag_gdrive.py

import os
import io
import streamlit as st
import numpy as np
from typing import List, Tuple
import json

# Libraries for reading files
import docx2txt
import PyPDF2

# Google auth
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.oauth2.service_account import Credentials

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
except:
    SentenceTransformer = None
try:
    import faiss
except:
    faiss = None
try:
    from mistralai import Mistral
except:
    Mistral = None


# ---------------- AUTH FUNCTION ---------------- #
def authenticate_gdrive():
    creds_json = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(
        creds_json,
        scopes=['https://www.googleapis.com/auth/drive']
    )

    gauth = GoogleAuth()
    gauth.credentials = creds
    drive = GoogleDrive(gauth)
    return drive


# ---------------- FETCH FUNCTION ---------------- #
def extract_text_from_file(drive_file):
    mime = drive_file['mimeType']

    if mime == "application/vnd.google-apps.document":
        txt = drive_file.GetContentString()
        return txt

    content = drive_file.GetContentFile(drive_file['title'])
    filename = drive_file['title']

    if filename.lower().endswith(".pdf"):
        with open(filename, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
    elif filename.lower().endswith(".docx"):
        text = docx2txt.process(filename)
    elif filename.lower().endswith(".txt"):
        text = drive_file.GetContentString()
    else:
        return None

    return text


def fetch_gdrive_files(drive, folder_id, max_files=15):
    query = f"'{folder_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()

    docs = []
    for f in file_list[:max_files]:
        text = extract_text_from_file(f)
        if text:
            docs.append((f['title'], text))

    return docs


# --------------- RAG UTILITIES ---------------- #
def chunk_text(text, chunk_size=600, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks


def compute_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("Install sentence-transformers")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, convert_to_numpy=True)
    return embs.astype(np.float32)


def build_faiss_index(embs):
    if faiss is None:
        raise RuntimeError("Install faiss-cpu")
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embs)
    return index


def query_index(index, q_emb, top_k):
    distances, indices = index.search(q_emb, top_k)
    return distances, indices


# ---------------- STREAMLIT UI ---------------- #
st.title("📑 Domain RAG — Google Drive")

with st.sidebar:
    st.header("⚙ Settings")
    MISTRAL_KEY = st.text_input("Mistral API Key", type="password")
    top_k = st.number_input("Top K", 1, 10, 4)
    chunk_size = st.number_input("Chunk Size", 200, 2000, 600)
    overlap = st.number_input("Overlap", 0, 400, 100)
    folder_id = st.text_input("Google Drive Folder ID")

    fetch_btn = st.button("📥 Fetch Docs")

if fetch_btn:
    try:
        drive = authenticate_gdrive()
        docs = fetch_gdrive_files(drive, folder_id)

        if not docs:
            st.error("❌ No readable docs found! Ensure sharing & file types (PDF/DOCX/TXT/Google Docs)")
        else:
            st.session_state["docs"] = docs
            st.success(f"Fetched {len(docs)} files ✔")
    except Exception as e:
        st.error(f"❌ {e}")


if st.button("⚡ Build Index"):
    if "docs" not in st.session_state:
        st.error("Fetch documents first")
    else:
        chunks, meta = [], []
        for title, text in st.session_state["docs"]:
            for i, c in enumerate(chunk_text(text, chunk_size, overlap)):
                chunks.append(c)
                meta.append({"title": title, "chunk": i})

        st.session_state["chunks"] = chunks
        st.session_state["meta"] = meta
        embs = compute_embeddings(chunks)
        st.session_state["embs"] = embs
        st.session_state["index"] = build_faiss_index(embs)

        st.success("Index built!")


st.header("🔍 Ask a Question")
query = st.text_input("Your query")
ask_btn = st.button("Ask")

if ask_btn:
    if "index" not in st.session_state:
        st.error("No index")
    else:
        q_emb = compute_embeddings([query])
        distances, idxs = query_index(st.session_state["index"], q_emb, top_k)

        answers = []
        for i in idxs[0]:
            st.write(f"📌 From: {st.session_state['meta'][i]['title']}")
            st.code(st.session_state["chunks"][i][:800])
