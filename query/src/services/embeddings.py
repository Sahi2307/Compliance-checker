from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class EmbeddingService:
    @staticmethod
    def create_embeddings():
        return HuggingFaceEmbeddings()

    @staticmethod
    def create_knowledge_base(chunks, embeddings):
        return FAISS.from_texts(chunks, embeddings)