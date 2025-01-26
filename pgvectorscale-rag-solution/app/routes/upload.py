from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.managers.embedding_manager import generate_and_store_embeddings  # Import the embedding function
import os

router = APIRouter()

# Ensure the 'uploads' directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, save it to the server, and process it to generate embeddings.
    """
    # Save file to the uploads directory
    file_location = f"{UPLOAD_DIR}/{file.filename}"

    try:
        with open(file_location, "wb") as f:
            f.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    # Call embedding manager to process the file
    try:
        result = generate_and_store_embeddings(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

    # Return success response
    return JSONResponse(
        content={
            "message": f"File '{file.filename}' uploaded and processed successfully!",
            "details": result,
        }
    )
