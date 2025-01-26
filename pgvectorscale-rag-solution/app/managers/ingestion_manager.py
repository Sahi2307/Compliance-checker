 #ingestion_manager
import os
from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import Dict

ALLOWED_EXTENSIONS = {".pdf"}

def save_file(file: UploadFile, upload_dir: str) -> str:
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path

def extract_text_from_pdf(file_path: str) -> list:
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if isinstance(docs, list) and all(isinstance(doc, Document) for doc in docs):
        return docs
    else:
        raise ValueError(f"Unexpected item type in docs: {type(docs)}.")

async def process_pdf(file: UploadFile, upload_dir: str) -> Dict:
    # Save the file and extract the text
    file_path = save_file(file, upload_dir)
    extracted_text = extract_text_from_pdf(file_path)
    
    return {"file_path": file_path, "text": extracted_text}

