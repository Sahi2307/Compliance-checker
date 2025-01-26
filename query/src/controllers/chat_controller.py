import streamlit as st
from src.services.pdf_processor import PDFProcessor
from src.services.embeddings import EmbeddingService
from src.services.vector_storage import VectorStorageService
from src.managers.model_manager import ModelManager

class ChatController:
    @staticmethod
    def run():
        st.set_page_config(page_title="Ask About Contract PDF")
        st.header("Ask About Contract PDF  🤓")

        vector_storage = VectorStorageService()

        pdf = st.file_uploader("Upload your PDF", type="pdf")
        
        if pdf is not None:
            with open(pdf.name, "wb") as f:
                f.write(pdf.getvalue())
            
            embeddings = EmbeddingService.create_embeddings()
            result = vector_storage.process_and_store_embeddings(
                pdf.name, 
                PDFProcessor.extract_text, 
                PDFProcessor.split_text,
                embeddings
            )
            st.write(result['message'])

            chunks = PDFProcessor.split_text(PDFProcessor.extract_text(pdf))
            knowledge_base = EmbeddingService.create_knowledge_base(chunks, embeddings)
            
            similar_chunks = []
            user_question = st.text_input("Ask a question about the PDF:")
            
            if user_question:
                similar_chunks = knowledge_base.similarity_search(user_question)
                k = 3
                top_k_chunks = similar_chunks[:k]
                
                _, chain = ModelManager.setup_model()
                response = chain.run(input_documents=top_k_chunks, question=user_question)
                
                k_values = [len(str(chunk)) for chunk in top_k_chunks]
                
                st.subheader("Similarity Search Results:")
                for i, chunk in enumerate(similar_chunks):
                    st.write(f"Chunk {i + 1}:", chunk)
                
                st.subheader(f"Top {k} Chunks Similar to the Question:")
                for i, chunk in enumerate(top_k_chunks):
                    st.write(f"Chunk {i + 1}:", chunk)
                
                st.subheader("Answer from LLM:")
                st.write(response)
                
                st.subheader("Determine 'k' value for each chunk retrieval:")
                for i, k_value in enumerate(k_values):
                    st.write(f"Chunk {i + 1}: {k_value}")