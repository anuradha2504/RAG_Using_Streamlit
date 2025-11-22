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


# ----------------------- Google Drive Auth -----------------------
def authenticate_gdrive():
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    service_account_info = st.secrets["gcp_service_account"]

    creds = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scopes)

    gauth = GoogleAuth()
    gauth.credentials = creds
    drive = GoogleDrive(gauth)
    return drive


# ----------------------- Fetch Files -----------------------
def fetch_gdrive_files(drive, folder_id: str, max_files=10):
    """Fetch docs & extract text from supported formats."""
    query = f"'{folder_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()

    docs = []

    for f in file_list[:max_files]:
        name = f['title']
        mime = f['mimeType']

        try:
            # Google Docs → convert to plain text
            if mime == "application/vnd.google-apps.document":
                content = f.GetContentString()
                docs.append((name, content))
                continue

            # Plain text / CSV / JSON
            if mime.startswith("text/") or mime == "application/json":
                content = f.GetContentString()
                docs.append((name, content))
                continue

            # PDF, DOCX fallback
            try:
                content = f.GetContentString(mimetype='text/plain')
                if content.strip():
                    docs.append((name, content))
            except:
                pass

        except Exception as e:
            print(f"⚠️ Skipped {name}: {e}")

    return docs


# ----------------------- Chunking -----------------------
def chunk_text(text, chunk_size=600, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks


# ------------------ Embedding + Index ------------------
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


def query_index(index, emb, top_k=4):
    dists, idxs = index.search(emb, top_k)
    return dists, idxs


def call_mistral(api_key, prompt, model="mistral-small-latest"):
    if Mistral is None:
        raise RuntimeError("Install mistralai")

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]


# ----------------------- UI -----------------------
st.set_page_config(page_title="Google Drive RAG", layout="wide")
st.title("📚 RAG from Google Drive — Mistral AI")


with st.sidebar:
    MISTRAL_KEY = st.text_input("🔑 Mistral API Key", type="password")
    folder_id = st.text_input("📂 Google Drive Folder ID")
    top_k = st.slider("Top-K results", 1, 8, 4)
    gdrive_fetch = st.button("📥 Fetch Google Drive Documents")


# -------------------- Fetch & Display Docs --------------------
if gdrive_fetch:
    try:
        drive = authenticate_gdrive()
        docs = fetch_gdrive_files(drive, folder_id)

        if len(docs) == 0:
            st.error("❌ No readable docs found! Add Google Docs / PDF / TXT / DOCX etc.")
        else:
            st.success(f"📄 Loaded {len(docs)} document(s)")
            for name, _ in docs:
                st.write(f"➤ {name}")
            st.session_state["docs"] = docs

    except Exception as e:
        st.error(f"Google Drive Error: {e}")


# -------------------- Indexing Btn --------------------
if st.button("⚡ Build Vector Index"):
    if "docs" not in st.session_state:
        st.error("Please fetch docs first.")
    else:
        docs = st.session_state["docs"]
        chunks, meta = [], []

        for title, text in docs:
            for i, c in enumerate(chunk_text(text)):
                chunks.append(c)
                meta.append({"title": title, "chunk": i})

        try:
            embs = compute_embeddings(chunks)
            index = build_faiss_index(embs)
            st.session_state.update({"embs": embs, "index": index,
                                    "chunks": chunks, "meta": meta})
            st.success(f"🎯 Indexed {len(chunks)} text chunks successfully!")

        except Exception as e:
            st.error(f"Embedding / Indexing Error: {e}")


# -------------------- Query Section --------------------
st.subheader("🔍 Ask a Question")
query = st.text_input("Type your question")

if st.button("▶️ Search & Answer"):
    if "index" not in st.session_state:
        st.error("Build index first!")
    elif not query:
        st.warning("Enter a question")
    else:
        q_emb = compute_embeddings([query])
        dists, idxs = query_index(st.session_state["index"], q_emb, top_k)

        chunks = st.session_state["chunks"]
        meta = st.session_state["meta"]
        context = []

        for i, idx in enumerate(idxs[0]):
            src = meta[idx]["title"]
            chk = meta[idx]["chunk"]
            st.markdown(f"**Match {i+1}: {src} (chunk {chk})**")
            snippet = chunks[idx]
            context.append(snippet)
            st.code(snippet[:500])

        if MISTRAL_KEY.strip():
            prompt = (
                "Use only this context:\n\n" +
                "\n\n---\n\n".join(context) +
                f"\n\nUser Question: {query}\nAnswer:"
            )
            try:
                answer = call_mistral(MISTRAL_KEY, prompt)
                st.subheader("💡 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Mistral API Error: {e}")
        else:
            st.warning("Provide Mistral Key for AI answer — showing context only")
            st.write("\n\n".join(context))
