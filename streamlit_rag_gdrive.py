# streamlit_rag_gdrive.py
# Streamlit app: Domain-specific RAG using MistralAI with Google Drive document ingestion (Service Account Auth)

import os
import streamlit as st
import numpy as np
from typing import List, Tuple
import json

# ----------------------- Optional Imports -----------------------
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from mistralai import Mistral
except Exception:
    Mistral = None

try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:
    GoogleAuth = None
    GoogleDrive = None
    ServiceAccountCredentials = None

# ----------------------- Utilities -----------------------

def authenticate_gdrive():
    """Authenticate Google Drive using service account credentials from Streamlit secrets."""
    if GoogleAuth is None or ServiceAccountCredentials is None:
        raise RuntimeError("PyDrive or oauth2client not installed. Please install them.")

    scopes = ['https://www.googleapis.com/auth/drive']
    # Read service account credentials from Streamlit secrets
    service_account_info = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scopes)

    gauth = GoogleAuth()
    gauth.credentials = creds
    drive = GoogleDrive(gauth)
    return drive

def fetch_gdrive_files(drive, folder_id: str, max_files=10) -> List[Tuple[str, str]]:
    """Fetch up to max_files text files from a given Google Drive folder."""
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    docs = []
    for f in file_list[:max_files]:
        if f['mimeType'] == 'text/plain':
            content = f.GetContentString()
            docs.append((f['title'], content))
    return docs

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks

def compute_embeddings_sbert(texts, model_name="all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("Install sentence-transformers to compute embeddings.")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs.astype(np.float32)

def build_faiss_index(embs):
    if faiss is None:
        raise RuntimeError("Install faiss-cpu to use vector store.")
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embs)
    return index

def query_index(index, q_emb, top_k=4):
    distances, indices = index.search(q_emb, top_k)
    return distances, indices

def call_mistral(api_key, prompt, model="mistral-medium"):
    if Mistral is None:
        raise RuntimeError("Install mistralai to use Mistral API.")
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers using provided sources. If answer is not present, say you don't know."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

# ----------------------- Streamlit UI -----------------------

st.set_page_config(page_title="Domain-specific RAG (GDrive + Mistral)", layout="wide")
st.title("📑 Domain-specific RAG — Google Drive + MistralAI")

with st.sidebar:
    st.header("⚙️ Configuration")
    MISTRAL_KEY = st.text_input("Mistral API Key", type="password")
    top_k = st.number_input("Top-k retrieved chunks", min_value=1, max_value=10, value=4)
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=600)
    overlap = st.number_input("Chunk overlap", min_value=0, max_value=400, value=100)
    folder_id = st.text_input("Google Drive Folder ID")
    gdrive_fetch = st.button("📥 Fetch Documents from Google Drive")

# --------- Fetch Docs ---------
if gdrive_fetch:
    try:
        drive = authenticate_gdrive()
        raw_docs = fetch_gdrive_files(drive, folder_id)
        if len(raw_docs) < 4:
            st.error("Less than 4 documents found. Add more files to the folder.")
        else:
            st.success(f"✅ Fetched {len(raw_docs)} documents from Google Drive.")
            st.session_state['raw_docs'] = raw_docs
    except Exception as e:
        st.error(f"❌ Google Drive fetch failed: {e}")

# --------- Build Index ---------
if st.button("⚡ Ingest and Build Index"):
    if 'raw_docs' not in st.session_state or len(st.session_state['raw_docs']) < 4:
        st.error("Please fetch at least 4 documents first.")
    else:
        raw_docs = st.session_state['raw_docs']
        chunks = []
        metadata = []
        for doc_id, text in raw_docs:
            for i, c in enumerate(chunk_text(text, chunk_size, overlap)):
                chunks.append(c)
                metadata.append({"source": doc_id, "chunk": i})
        st.session_state['chunks'] = chunks
        st.session_state['metadata'] = metadata

        try:
            embs = compute_embeddings_sbert(chunks)
            st.session_state['embs'] = embs
            index = build_faiss_index(embs)
            st.session_state['index'] = index
            st.success(f"✅ Built FAISS index with {len(chunks)} chunks.")
        except Exception as e:
            st.error(f"❌ Embedding/Indexing error: {e}")

# --------- Query Section ---------
st.header("🔎 Query")
query = st.text_input("Enter your question")

if st.button("▶️ Run Query"):
    if 'index' not in st.session_state:
        st.error("No index found. Fetch and ingest documents first.")
    elif not query:
        st.error("Please enter a query.")
    else:
        q_emb = compute_embeddings_sbert([query])
        distances, indices = query_index(st.session_state['index'], q_emb, top_k)
        chunks = st.session_state['chunks']
        metadata = st.session_state['metadata']
        context_texts = []
        for rank, hit in enumerate(indices[0]):
            snippet = chunks[hit]
            context_texts.append(f"Source: {metadata[hit]['source']} — {snippet}")
            st.markdown(f"**Rank {rank+1} — Source:** {metadata[hit]['source']} (chunk {metadata[hit]['chunk']})")
            st.code(snippet[:800])

        prompt = "Use these sources to answer the question. Cite source numbers.\n\n"
        for i, ctx in enumerate(context_texts):
            prompt += f"[{i+1}] {ctx}\n\n"
        prompt += f"User Question: {query}\nAnswer:"

        if MISTRAL_KEY:
            try:
                with st.spinner("🤖 Calling MistralAI..."):
                    answer = call_mistral(MISTRAL_KEY, prompt)
                st.subheader("💡 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Mistral API error: {e}")
        else:
            st.warning("⚠️ No Mistral API key provided — showing retrieved context only.")
            st.subheader("Retrieved Context")
            st.write("\n\n".join(context_texts))
