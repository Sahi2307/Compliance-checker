import os
import pickle
import time
from pinecone import Pinecone, ServerlessSpec
from src.config.settings import settings

class VectorStorageService:
    def _init_(self, 
                 pinecone_api_key=settings.PINECONE_API_KEY, 
                 pinecone_environment=settings.PINECONE_ENVIRONMENT, 
                 pinecone_index_name=settings.PINECONE_INDEX_NAME):
        self.pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
        self.index_name = pinecone_index_name
        self.vectorstore_dir = "vectorstores"
        os.makedirs(self.vectorstore_dir, exist_ok=True)

    def store_locally(self, embeddings, chunks, metadata, filename="local_vectors.pkl"):
        file_path = os.path.join(self.vectorstore_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        local_data = {
            "embeddings": embeddings,
            "chunks": chunks,
            "metadata": metadata
        }
        
        with open(file_path, "wb") as f:
            pickle.dump(local_data, f)

    def store_in_pinecone(self, embeddings, chunks, metadata, pdf_filename):
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
        
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=len(embeddings[0]),
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        index = self.pc.Index(self.index_name)
        index.upsert(vectors=vectors, namespace=unique_namespace)

    def process_and_store_embeddings(self, file_path, text_extractor, chunk_splitter, embeddings):
        try:
            docs = text_extractor(file_path)
            chunks = chunk_splitter(docs)
            
            metadata = [
                {
                    "filename": os.path.basename(file_path),
                    "page": chunk.get("metadata", {}).get('page', 0) if isinstance(chunk, dict) else chunk.metadata.get('page', 0),
                    "chunk_index": idx
                }
                for idx, chunk in enumerate(chunks)
            ]
            
            self.store_locally(embeddings, chunks, metadata)
            self.store_in_pinecone(embeddings, chunks, metadata, file_path)
            
            return {"message": "Embeddings generated and stored successfully."}
        except Exception as e:
            print(f"Error occurred: {e}")
            return {"message": f"Failed to process the file: {e}"}