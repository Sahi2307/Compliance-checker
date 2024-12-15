import os
from dotenv import load_dotenv

load_dotenv()

PDF_UPLOAD_PATH = os.getenv("PDF_UPLOAD_PATH", "./data/uploads")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
