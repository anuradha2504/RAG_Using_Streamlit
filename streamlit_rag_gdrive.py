import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import io
import docx2txt
import PyPDF2
import tempfile


def authenticate_gdrive():
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    credentials = Credentials.from_service_account_file(
        'service_account.json',
        scopes=scopes
    )
    return build('drive', 'v3', credentials=credentials)


def extract_pdf(file_bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return ""


def extract_docx(file_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        return text
    except:
        return ""


def fetch_gdrive_files(drive, folder_id):
    query = f"'{folder_id}' in parents and trashed=false"
    results = drive.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])

    docs = []

    for f in items:
        file_id = f['id']
        name = f['name']
        mime = f['mimeType']

        try:
            # Google Docs Export
            if mime == "application/vnd.google-apps.document":
                request = drive.files().export_media(fileId=file_id,
                                                     mimeType='text/plain')
                file_bytes = request.execute()
                docs.append({"name": name, "text": file_bytes.decode("utf-8")})
            
            # Google Sheets Export
            elif mime == "application/vnd.google-apps.spreadsheet":
                request = drive.files().export_media(fileId=file_id,
                                                     mimeType="text/csv")
                file_bytes = request.execute()
                docs.append({"name": name, "text": file_bytes.decode("utf-8")})

            # Google Slides Export (as text)
            elif mime == "application/vnd.google-apps.presentation":
                request = drive.files().export_media(fileId=file_id,
                                                     mimeType="text/plain")
                file_bytes = request.execute()
                docs.append({"name": name, "text": file_bytes.decode("utf-8")})

            # PDF / DOCX / TXT Download
            else:
                request = drive.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                
                content = fh.getvalue()

                if mime == "application/pdf":
                    text = extract_pdf(content)
                elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_docx(content)
                elif mime == "text/plain":
                    text = content.decode("utf-8", errors="ignore")
                else:
                    text = ""  # unsupported

                if text.strip():
                    docs.append({"name": name, "text": text})

        except Exception as e:
            st.warning(f"⚠️ Failed reading {name}: {e}")

    return docs


# --------- Streamlit UI ---------
st.title("📂 Google Drive Document Loader")

folder_id = st.text_input("Enter Google Drive Folder ID")

if st.button("Fetch Files"):
    try:
        drive = authenticate_gdrive()
        docs = fetch_gdrive_files(drive, folder_id)

        if len(docs) == 0:
            st.error("❌ No readable docs found! Add Google Docs / PDF / TXT / DOCX etc.")
        else:
            st.success(f"✅ Fetched & extracted {len(docs)} readable documents!")
            st.session_state["raw_docs"] = docs
            for d in docs:
                st.write(f"📄 {d['name']} - {len(d['text'])} characters")

    except Exception as e:
        st.error(f"❌ Error: {e}")
