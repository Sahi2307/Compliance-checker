from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict

from src.utils.pdf_utils import load_pdf
from src.utils.text_splitter_utils import split_document
from src.utils.embedding_utils import create_embeddings
from src.utils.vectorstore_utils import store_locally, store_in_pinecone

router = APIRouter()

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict:
    upload_dir = "uploads"
    
    try:
        docs = load_pdf(file, upload_dir)
        chunks = split_document(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to split the document into chunks.")
        embeddings = create_embeddings(chunks)
        if not embeddings or len(embeddings) != len(chunks):
            raise HTTPException(status_code=500, detail="Embedding creation failed.")
        metadata = [
            {
                "filename": file.filename,
                "page": chunk.metadata.get('page', 0),
                "chunk_index": idx
            } 
            for idx, chunk in enumerate(chunks)
        ]
        store_locally(embeddings, chunks, metadata)
        store_in_pinecone(embeddings, chunks, metadata,file.filename)
        chunk_details = [
            {
                "chunk_index": idx,
                "chunk_size": len(chunk.page_content),
                "page": chunk.metadata.get('page', 0)
            }
            for idx, chunk in enumerate(chunks)
        ]
        
        return {
            "message": "PDF processed and embeddings created successfully.",
            "filename": file.filename,
            "chunks_count": len(chunks),
            "chunk_details": chunk_details,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the PDF: {str(e)}")
