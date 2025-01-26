from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_document(docs):
    """Split document text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks
