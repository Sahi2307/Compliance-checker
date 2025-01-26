from langchain_huggingface import HuggingFaceEmbeddings
from app.managers.ingestion_manager import extract_text_from_pdf
from app.managers.chunk_manager import split_document
import os
import pickle
from pinecone import Pinecone, ServerlessSpec
import time

PINECONE_API_KEY = "apikey"
PINECONE_ENVIRONMENT = "env"
PINECONE_INDEX_NAME = "contract"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

VECTORSTORE_DIR = "vectorstores"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

def create_embeddings(chunks):
    """
    Create embeddings from the chunk contents using a HuggingFace model.

    Args:
        chunks (list): List of chunk objects or dictionaries containing `page_content`.

    Returns:
        embeddings (list): List of embeddings corresponding to the chunk contents.
    """
    model_name = "BAAI/bge-large-en-v1.5"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Handle both objects and dictionaries
    chunk_contents = [chunk["page_content"] if isinstance(chunk, dict) else chunk.page_content for chunk in chunks]
    print(f"Chunks contents: {chunk_contents[:5]}")  # Debugging, first 5 chunks

    embeddings = hf_embeddings.embed_documents(chunk_contents)
    return embeddings

def store_locally(embeddings, chunks, metadata, filename="local_vectors.pkl"):
    """
    Store embeddings, chunks, and metadata locally in a pickle file.

    Args:
        embeddings (list): List of embeddings.
        chunks (list): List of chunk objects.
        metadata (list): List of metadata dictionaries.
        filename (str): The name of the file to store data in.
    """
    file_path = os.path.join(VECTORSTORE_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    print(f"Storing embeddings locally at {file_path}...")  # Debugging

    local_data = {
        "embeddings": embeddings,
        "chunks": chunks,
        "metadata": metadata
    }
    with open(file_path, "wb") as f:
        pickle.dump(local_data, f)

def store_in_pinecone(embeddings, chunks, metadata, pdf_filename):
    """
    Store embeddings in Pinecone with associated metadata.

    Args:
        embeddings (list): List of embeddings.
        chunks (list): List of chunk objects.
        metadata (list): List of metadata dictionaries.
        pdf_filename (str): Name of the original PDF file.
    """
    print(f"Storing embeddings in Pinecone...")  # Debugging
    base_name = os.path.splitext(pdf_filename)[0].replace(" ", "_").lower()
    unique_namespace = f"{base_name}-{int(time.time())}"

    vectors = [
        {
            "id": f"{unique_namespace}-chunk-{idx}",
            "values": embedding,
            "metadata": {**meta}
        }
        for idx, (embedding, chunk, meta) in enumerate(zip(embeddings, chunks, metadata))
    ]

    if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        print(f"Creating Pinecone index {PINECONE_INDEX_NAME}...")  # Debugging
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=len(embeddings[0]),
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )

    index = pc.Index(PINECONE_INDEX_NAME)
    index.upsert(vectors=vectors, namespace=unique_namespace)

def generate_and_store_embeddings(file_path):
    """
    Extract text from PDF, split it into chunks, generate embeddings, and store locally and in Pinecone.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        dict: A message indicating success or failure.
    """
    try:
        print(f"Extracting text from PDF: {file_path}")  # Debugging
        docs = extract_text_from_pdf(file_path)
        chunks = split_document(docs)

        print(f"Chunks: {chunks[:5]}")  # Debugging, first 5 chunks
        embeddings = create_embeddings(chunks)

        metadata = [
            {
                "filename": os.path.basename(file_path),
                "page": chunk.get("metadata", {}).get('page', 0) if isinstance(chunk, dict) else chunk.metadata.get('page', 0),
                "chunk_index": idx
            }
            for idx, chunk in enumerate(chunks)
        ]
        print(f"Metadata: {metadata[:2]}")  # Debugging, first 5 metadata entries

        store_locally(embeddings, chunks, metadata)
        store_in_pinecone(embeddings, chunks, metadata, file_path)

        return {"message": "Embeddings generated and stored successfully."}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"message": f"Failed to process the file: {e}"}
