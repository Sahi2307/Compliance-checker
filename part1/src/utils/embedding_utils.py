from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def create_embeddings(chunks):
    """Generate embeddings for text chunks."""
    model_name = "BAAI/bge-small-en"
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    chunk_contents = [chunk.page_content for chunk in chunks]
    embeddings = hf_embeddings.embed_documents(chunk_contents)  
    return embeddings
