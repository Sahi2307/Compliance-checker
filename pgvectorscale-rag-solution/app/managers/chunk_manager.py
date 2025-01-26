 # Chunk Manager: chunk_manager.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.managers.ingestion_manager import extract_text_from_pdf


def split_document(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

def split_and_process_chunks(file_path):
    docs = extract_text_from_pdf(file_path)
    chunks = split_document(docs)
    return {"message": "Chunks created successfully.", "chunks_count": len(chunks)}