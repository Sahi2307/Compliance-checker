import streamlit as st
import pandas as pd
import PyPDF2  # To extract content from PDFs
from io import BytesIO  # To handle in-memory file objects
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer, SynthesizedResponse

# Initialize the vector search (replace this with your actual setup)
vec = VectorStore()  # Adjust with the actual implementation

# Streamlit App Title
st.title("Legal Contract Query Assistant")
st.subheader("Analyze and answer questions about commercial legal contracts.")

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF contract:", type=["pdf"])

# Function to convert text to a PDF using ReportLab
def generate_pdf(response_text):
    # Create an in-memory file
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    paragraphs = []

    # Split response into paragraphs
    for line in response_text.split("\n"):
        if line.strip():
            paragraphs.append(Paragraph(line.strip(), styles['Normal']))

    doc.build(paragraphs)
    buffer.seek(0)
    return buffer

# Submit Button
if st.button("Analyze Contract"):
    if not uploaded_file:
        st.warning("Please upload a PDF file before submitting.")
    else:
        with st.spinner("Processing your contract..."):
            try:
                # Extract text from the uploaded PDF
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pdf_content = ""
                for page in pdf_reader.pages:
                    pdf_content += page.extract_text()

                # Ensure the PDF content is not empty
                if not pdf_content.strip():
                    st.error("Unable to extract text from the uploaded PDF. Please check the file.")
                else:
                    # Perform the vector search
                    results = vec.search(pdf_content, limit=3)

                    # Generate a response using the Synthesizer
                    response: SynthesizedResponse = Synthesizer.generate_response(
                        question=pdf_content, context=results
                    )

                    # Display the results
                    st.subheader("Synthesized Response")
                    st.write(response.answer)

                    # Generate PDF and provide download option
                    pdf_file = generate_pdf(response.answer)
                    st.download_button(
                        label="Download Response as PDF",
                        data=pdf_file,
                        file_name="contract_analysis_report.pdf",
                        mime="application/pdf",
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("Powered by **Streamlit**, **PyPDF2**, **ReportLab**, and **OpenAI**")

