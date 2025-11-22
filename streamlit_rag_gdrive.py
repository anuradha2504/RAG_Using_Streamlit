
"""

import os
import io
import json
import tempfile
from typing import List, Dict, Tuple

import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Text extraction
import docx2txt
import PyPDF2

# Embeddings and FAISS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# For calling HuggingFace Inference API (optional)
import requests

# ---------------------------- Configuration ----------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_meta.json"

# ---------------------------- Utils ----------------------------

def build_drive_service(gcp_secret: Dict) -> "googleapiclient.discovery.Resource":
    # Expecting a dict structured as a service account JSON
    creds = Credentials.from_service_account_info(gcp_secret, scopes=[
        'https://www.googleapis.com/auth/drive.readonly'
    ])
    service = build('drive', 'v3', credentials=creds, cache_discovery=False)
    return service


def list_files_in_folder(service, folder_id: str) -> List[Dict]:
    query = f"'{folder_id}' in parents and trashed=false"
    results = []
    page_token = None
    while True:
        resp = service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name, mimeType)', pageToken=page_token).execute()
        results.extend(resp.get('files', []))
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return results


def download_file(service, file_id: str, file_name: str) -> bytes:
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return fh.getvalue()


def extract_text_from_bytes(content_bytes: bytes, mime_type: str, filename: str) -> str:
    if mime_type == 'application/pdf' or filename.lower().endswith('.pdf'):
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
            text = []
            for p in reader.pages:
                text.append(p.extract_text() or "")
            return "\n".join(text)
        except Exception as e:
            st.warning(f"PDF parsing failed for {filename}: {e}")
            return ""
    elif filename.lower().endswith('.docx') or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(content_bytes)
            tmp.flush()
            text = docx2txt.process(tmp.name)
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return text or ""
    else:
        # treat as text
        try:
            return content_bytes.decode('utf-8', errors='ignore')
        except Exception:
            return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

# ---------------------------- Embedding and FAISS helpers ----------------------------

@st.cache_resource
def load_embedding_model(model_name: str = EMB_MODEL_NAME):
    return SentenceTransformer(model_name)


def create_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    return embs.astype('float32')


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def save_faiss(index, path: str):
    faiss.write_index(index, path)


def load_faiss(path: str):
    return faiss.read_index(path)

# ---------------------------- LLM Backends (Mistral via HF or OpenAI) ----------------------------

class LLMWrapper:
    def __init__(self, hf_token: str = None, openai_key: str = None):
        self.hf_token = hf_token
        self.openai_key = openai_key

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        # Prefer HF (Mistral) if token present
        if self.hf_token:
            return self._hf_complete(prompt, max_tokens)
        elif self.openai_key:
            return self._openai_complete(prompt, max_tokens)
        else:
            raise RuntimeError('No LLM API key provided. Set HF token or OpenAI key in streamlit.secrets')

    def _hf_complete(self, prompt: str, max_tokens: int = 512) -> str:
        # Use HuggingFace Inference API - text generation endpoint
        HF_URL = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct'  # example
        headers = {
            'Authorization': f'Bearer {self.hf_token}',
            'Accept': 'application/json'
        }
        payload = {
            'inputs': prompt,
            'parameters': { 'max_new_tokens': max_tokens, 'return_full_text': False }
        }
        resp = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            j = resp.json()
            if isinstance(j, list) and 'generated_text' in j[0]:
                return j[0]['generated_text']
            # some endpoints return {'generated_text': ...}
            if isinstance(j, dict) and 'generated_text' in j:
                return j['generated_text']
            # fallback
            return str(j)
        else:
            raise RuntimeError(f"HF inference failed: {resp.status_code} {resp.text}")

    def _openai_complete(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            import openai
        except Exception:
            raise RuntimeError('openai package missing; install openai to use OpenAI backend')
        openai.api_key = self.openai_key
        resp = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
        )
        return resp['choices'][0]['message']['content']

# ---------------------------- Streamlit App ----------------------------

st.set_page_config(page_title='Healthcare RAG Bot (Drive -> FAISS -> Mistral)', layout='wide')
st.title('Healthcare RAG Bot — Drive + FAISS + Mistral')

# Load secrets
gcp_secret = st.secrets.get('gcp_service_account')
hf_token = st.secrets.get('hf', {}).get('api_token') if st.secrets.get('hf') else None
openai_key = st.secrets.get('openai', {}).get('api_key') if st.secrets.get('openai') else None

if not gcp_secret:
    st.error('Google service account not found in streamlit.secrets["gcp_service_account"]. Add it first.')
    st.stop()

service = build_drive_service(gcp_secret)

with st.sidebar:
    st.header('Drive settings')
    folder_id = st.text_input('Google Drive folder ID', value='')
    if st.button('List files in folder') and folder_id:
        try:
            files = list_files_in_folder(service, folder_id)
            st.write(f'Found {len(files)} files')
            for f in files:
                st.write(f"- {f['name']} ({f['mimeType']}) — id: {f['id']}")
        except Exception as e:
            st.error(f'Error listing files: {e}')

    st.header('Index controls')
    if st.button('Rebuild index from Drive'):
        if not folder_id:
            st.warning('Please enter folder ID first')
        else:
            with st.spinner('Downloading files and building index...'):
                files = list_files_in_folder(service, folder_id)
                docs = []
                meta = []
                for f in files:
                    try:
                        raw = download_file(service, f['id'], f['name'])
                        text = extract_text_from_bytes(raw, f.get('mimeType',''), f['name'])
                        if not text.strip():
                            continue
                        chunks = chunk_text(text)
                        for i, c in enumerate(chunks):
                            docs.append(c)
                            meta.append({'source': f['name'], 'file_id': f['id'], 'chunk': i})
                    except Exception as e:
                        st.warning(f"Failed to process {f['name']}: {e}")

                if not docs:
                    st.warning('No textual documents found to index.')
                else:
                    emb_model = load_embedding_model()
                    embeddings = create_embeddings(emb_model, docs)
                    # normalize for cosine similarity in IP index
                    faiss.normalize_L2(embeddings)
                    index = build_faiss_index(embeddings)
                    save_faiss(index, FAISS_INDEX_PATH)
                    with open(METADATA_PATH, 'w', encoding='utf-8') as fo:
                        json.dump(meta, fo, ensure_ascii=False, indent=2)
                    st.success(f'Indexed {len(docs)} chunks. FAISS saved to {FAISS_INDEX_PATH}')

# Load index if exists
index = None
meta = None
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
    try:
        index = load_faiss(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'r', encoding='utf-8') as fi:
            meta = json.load(fi)
    except Exception as e:
        st.warning(f'Failed to load existing index: {e}')

llm = LLMWrapper(hf_token=hf_token, openai_key=openai_key)
emb_model = load_embedding_model()

# Chat UI
query = st.text_area('Ask the healthcare bot a question (clinical, process, policy, non-diagnostic):', height=120)
if st.button('Ask') and query.strip():
    if index is None or meta is None:
        st.error('No FAISS index found. Click "Rebuild index from Drive" in the sidebar first.')
    else:
        # embed query
        q_emb = create_embeddings(emb_model, [query])
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, k=5)
        retrieved_texts = []
        for idx in I[0]:
            if idx < len(meta):
                # load chunk text from metadata by re-reading files or store texts on disk; here we assume in-memory not kept
                # For simplicity we stored chunk texts as part of metadata in production store them in DB or a file
                retrieved_texts.append(f"[source={meta[idx]['source']} chunk={meta[idx]['chunk']}]")
            else:
                retrieved_texts.append('[unknown]')

        # Compose prompt for the LLM
        context = '\n\n'.join(retrieved_texts)
        prompt = ("You are a helpful healthcare assistant for internal use. Use the provided retrieved passages as context. "
                  "Answer concisely and cite sources in square brackets. Do NOT provide medical diagnosis — only provide educational and process-level guidance.\n\n"
                  f"Context:\n{context}\n\nUser Question:\n{query}\n\nAnswer:")
        with st.spinner('Generating answer from LLM...'):
            try:
                answer = llm.complete(prompt, max_tokens=512)
                st.markdown('**Answer:**')
                st.write(answer)
                st.markdown('**Retrieved sources:**')
                for r in retrieved_texts:
                    st.write(r)
            except Exception as e:
                st.error(f'LLM call failed: {e}')

# End of file
