#A pyhton script which will take 3 pdfs as input will output corresponding analysed pdfs.
import streamlit as st
import PyPDF2  # To extract content from PDFs
from io import BytesIO  # To handle in-memory file objects
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer, SynthesizedResponse
# Removed tiktoken dependency - using Google Gemini instead

# Initialize the vector search (replace this with your actual setup)
vec = VectorStore()  # Adjust with the actual implementation

# Streamlit App Title
st.title("Legal Contract Query Assistant")

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload up to 3 PDF contracts:",
    type=["pdf"],
    accept_multiple_files=True,
)

# Optional: Ask a question about the uploaded PDFs
user_question = st.text_input(
    "Ask a question about the uploaded PDFs:",
    placeholder="e.g., What is the governing law?",
)

# Function to count tokens - simplified for Google Gemini
def count_tokens(text):
    # Simple word-based token estimation (rough approximation)
    return len(text.split()) * 1.3  # Rough estimate: 1.3 tokens per word

# Enhanced PDF Generation Function
def generate_pdf_with_features(response_text, uploaded_pdf_name):
    """
    Generate a styled PDF based on input text, including bold, normal text, and bullet points.
    The uploaded PDF's name (without extension) is displayed as the title of the output PDF.
    """
    # Remove the .pdf extension from the uploaded file name
    pdf_name = uploaded_pdf_name.rsplit(".", 1)[0]
    title_text = f"{pdf_name} Analysis Report"

    # Create an in-memory file
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Define styles
    normal_style = styles['Normal']
    bold_style = styles['Heading1']
    title_style = styles['Title']

    # Create a list to hold Paragraph objects
    paragraphs = []

    # Add title
    paragraphs.append(Paragraph(title_text, title_style))
    paragraphs.append(Spacer(1, 20))  # Add spacing after the title

    # Process the response_text
    lines = response_text.split('\n')
    for line in lines:
        if line.startswith('**') and line.endswith('**'):
            # Bold text (heading-like)
            paragraphs.append(Paragraph(line.strip('**'), bold_style))
            paragraphs.append(Spacer(1, 12))  # Add spacing after headings
        elif line.startswith('- '):
            # Bullet points
            paragraphs.append(Paragraph(line, normal_style))
        else:
            # Normal text
            paragraphs.append(Paragraph(line, normal_style))
            paragraphs.append(Spacer(1, 10))  # Add spacing after paragraphs

    # Build the PDF
    doc.build(paragraphs)
    buffer.seek(0)
    return buffer

# Initialize session state to store results
if "pdf_responses" not in st.session_state:
    st.session_state.pdf_responses = []
if "qa_responses" not in st.session_state:
    st.session_state.qa_responses = []

# Process Button
if st.button("Analyze Contracts"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file before submitting.")
    else:
        total_tokens = 0
        pdf_responses = []

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Extract text from the uploaded PDF
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    pdf_content = ""
                    for page in pdf_reader.pages:
                        pdf_content += page.extract_text()

                    # Ensure the PDF content is not empty
                    if not pdf_content.strip():
                        st.error(f"Unable to extract text from {uploaded_file.name}. Please check the file.")
                        continue

                    # Count tokens for the current file
                    token_count = count_tokens(pdf_content)
                    total_tokens += token_count

                    # Check token limit (adjusted for Google Gemini)
                    if total_tokens > 1000000:  # Gemini has much higher token limits
                        st.error(
                            f"Total token limit exceeded! The combined content of uploaded PDFs exceeds 1M tokens."
                        )
                        break

                    # Perform the vector search (top-1 only)
                    results = vec.search(pdf_content, limit=1)

                    # Generate a response using the Synthesizer
                    response: SynthesizedResponse = Synthesizer.generate_response(
                        question=pdf_content, context=results
                    )

                    # Store the response and file name for generating the PDF
                    pdf_responses.append((response.answer, uploaded_file.name))

                except Exception as e:
                    st.error(f"An error occurred while processing {uploaded_file.name}: {e}")

        # Save the responses in session state
        st.session_state.pdf_responses = pdf_responses

# Q&A across uploaded PDFs (top-1 retrieval per PDF)
if st.button("Ask Question for PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
    elif not user_question.strip():
        st.warning("Please enter a question.")
    else:
        qa_outputs = []
        for uploaded_file in uploaded_files:
            with st.spinner(f"Answering question for {uploaded_file.name}..."):
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    pdf_content = ""
                    for page in pdf_reader.pages:
                        pdf_content += page.extract_text()
                    if not pdf_content.strip():
                        st.error(f"Unable to extract text from {uploaded_file.name}. Skipping.")
                        continue
                    # Retrieve top-1 relevant chunk
                    results = vec.search(pdf_content, limit=1)
                    # Generate answer using the question + retrieved context
                    response: SynthesizedResponse = Synthesizer.generate_response(
                        question=user_question, context=results
                    )
                    # Save report file
                    reports_dir = Path(__file__).resolve().parent.parent / "reports"
                    reports_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_name = f"{uploaded_file.name.rsplit('.', 1)[0]}_qa_{timestamp}.pdf"
                    pdf_file = generate_pdf_with_features(response.answer, uploaded_file.name)
                    (reports_dir / out_name).write_bytes(pdf_file.getbuffer())
                    qa_outputs.append((uploaded_file.name, response.answer, out_name, pdf_file))
                except Exception as e:
                    st.error(f"An error occurred while answering for {uploaded_file.name}: {e}")
        st.session_state.qa_responses = qa_outputs

# Display Download Buttons
if "pdf_responses" in st.session_state and st.session_state.pdf_responses:
    st.subheader("Download Analysis Reports:")
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for response_text, file_name in st.session_state.pdf_responses:
        pdf_file = generate_pdf_with_features(response_text, file_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"{file_name.rsplit('.', 1)[0]}_analysis_{timestamp}.pdf"
        (reports_dir / out_name).write_bytes(pdf_file.getbuffer())
        st.download_button(
            label=f"Download Analysis for {file_name}",
            data=pdf_file,
            file_name=out_name,
            mime="application/pdf",
        )

# Display Q&A results and downloads
if "qa_responses" in st.session_state and st.session_state.qa_responses:
    st.subheader("Q&A Reports:")
    for file_name, answer_text, out_name, pdf_file in st.session_state.qa_responses:
        st.markdown(f"**{file_name}**")
        st.write(answer_text)
        st.download_button(
            label=f"Download Q&A for {file_name}",
            data=pdf_file,
            file_name=out_name,
            mime="application/pdf",
        )
