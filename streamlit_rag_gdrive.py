from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os

# Google Drive API Scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    creds = Credentials.from_service_account_file("service_account.json", scopes=SCOPES)

    # FIX for: Credentials object has no attribute 'access_token_expired'
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return build('drive', 'v3', credentials=creds)

def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return results.get("files", [])

def download_file(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {file_name}: {int(status.progress() * 100)}%")

    print(f"Downloaded: {file_name}")

if __name__ == "__main__":
    folder_id = "1TrTPnEefhmWQYZOOBOFTTtWD4ToT3ehp"  # your drive folder id
    
    service = get_drive_service()
    files = list_files_in_folder(service, folder_id)
    
    if not files:
        print("No files found.")
    else:
        print("Files found:")
        for file in files:
            print(file["name"])
            download_file(service, file["id"], file["name"])
