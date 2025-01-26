from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_file):
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def split_text(text, chunk_size=1000, chunk_overlap=200):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return text_splitter.split_text(text)