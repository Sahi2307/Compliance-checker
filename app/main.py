import streamlit as st
import pandas as pd
import PyPDF2  # To extract content from PDFs
from io import BytesIO  # To handle in-memory file objects
from pathlib import Path
from datetime import datetime
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

# Keep extracted text in session for follow-up questions
if "pdf_content" not in st.session_state:
	st.session_state.pdf_content = ""

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

user_question = st.text_input(
	"Ask a question about the uploaded contract:",
	placeholder="e.g., What is the governing law?",
)

# Extract on upload (once)
if uploaded_file and not st.session_state.pdf_content:
	try:
		pdf_reader = PyPDF2.PdfReader(uploaded_file)
		extracted = []
		for page in pdf_reader.pages:
			text = page.extract_text() or ""
			extracted.append(text)
		st.session_state.pdf_content = "\n".join(extracted)
	except Exception as e:
		st.error(f"Failed to extract text from PDF: {e}")

# Q&A Button
if st.button("Ask"):
	if not uploaded_file:
		st.warning("Please upload a PDF file first.")
	elif not user_question.strip():
		st.warning("Please enter a question.")
	elif not st.session_state.pdf_content.strip():
		st.error("No text could be extracted from the PDF.")
	else:
		with st.spinner("Answering your question..."):
			try:
				# Retrieve top-1 relevant chunk from vector DB using the PDF content as context seed
				results = vec.search(st.session_state.pdf_content, limit=1)

				# Generate answer using question + retrieved context
				response: SynthesizedResponse = Synthesizer.generate_response(
					question=user_question, context=results
				)

				st.subheader("Answer")
				st.write(response.answer)

				# Optional download of the answer as PDF
				pdf_file = generate_pdf(response.answer)
				# Save to reports folder
				reports_dir = Path(__file__).resolve().parent.parent / "reports"
				reports_dir.mkdir(parents=True, exist_ok=True)
				filename = f"contract_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
				(out_path := reports_dir / filename).write_bytes(pdf_file.getbuffer())
				st.download_button(
					label="Download Answer as PDF",
					data=pdf_file,
					file_name=filename,
					mime="application/pdf",
				)
			except Exception as e:
				st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("Powered by **Streamlit**, **PyPDF2**, **ReportLab**, and **Google Gemini**")

