import os
import pickle
from pinecone import Pinecone, ServerlessSpec  
import time 

PINECONE_API_KEY = "your-api-key-here"
PINECONE_ENVIRONMENT = "your-pinecone-environment-key"  
PINECONE_INDEX_NAME = "Your Pinecone index name"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

VECTORSTORE_DIR = "vectorstores"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

embedding_file = os.path.join(VECTORSTORE_DIR, "local_vectors.pkl")

def store_locally(embeddings, chunks, metadata, filename="local_vectors.pkl"):
    """Store the embeddings and metadata locally."""
    file_path = os.path.join(VECTORSTORE_DIR, filename)
    
    if os.path.exists(file_path):
        print(f"Removing old embeddings file at {file_path}...")
        os.remove(file_path) 

    local_data = {
        "embeddings": embeddings,
        "chunks": chunks,
        "metadata": metadata
    }
    with open(file_path, "wb") as f:
        pickle.dump(local_data, f)
    print(f"Local vector store saved at {file_path}")

def store_in_pinecone(embeddings, chunks, metadata, pdf_filename):
    """Storing embeddings in Pinecone using a unique namespace."""
    base_name = os.path.splitext(pdf_filename)[0].replace(" ", "_").lower()
    unique_namespace = f"{base_name}-{int(time.time())}"  

    vectors = [
        {
            "id": f"{unique_namespace}-chunk-{idx}",
            "values": embedding,
            "metadata": {"text": chunk.page_content, **meta}
        }
        for idx, (embedding, chunk, meta) in enumerate(zip(embeddings, chunks, metadata))
    ]

    try:
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"Creating Pinecone index {PINECONE_INDEX_NAME}...")
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
        print(f"Stored {len(vectors)} vectors in Pinecone under namespace '{unique_namespace}'.")

    except Exception as e:
        print(f"Error storing vectors in Pinecone: {e}")
        raise
