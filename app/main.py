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
st.title("Legal Contract Assistant")
st.subheader("Analyze and answer questions about commercial legal contracts.")

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF contract:", type=["pdf"])

# Keep extracted text in session for follow-up questions
if "pdf_content" not in st.session_state:
	st.session_state.pdf_content = ""

# Function to convert text to a styled PDF with a title using ReportLab
def generate_pdf_with_features(response_text, uploaded_pdf_name, report_title: str = "Q&A Report"):
	# Create an in-memory file
	buffer = BytesIO()
	doc = SimpleDocTemplate(buffer, pagesize=letter)
	styles = getSampleStyleSheet()

	# Derive a nice title from the uploaded file name
	pdf_name = uploaded_pdf_name.rsplit(".", 1)[0] if uploaded_pdf_name else "Contract"
	title_text = f"{pdf_name} {report_title}"

	# Build paragraphs
	paragraphs = []
	paragraphs.append(Paragraph(title_text, styles['Title']))
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
		text_joined = "\n".join(extracted)
		st.session_state.pdf_content = text_joined
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
			# Attempt direct structured answer from '-Answer' fields first (no spinner)
			_q_lower = user_question.strip().lower()
			label_map = {
				"governing law": "Governing Law-Answer",
				"renewal term": "Renewal Term-Answer",
				"effective date": "Effective Date-Answer",
				"expiration date": "Expiration Date-Answer",
				"parties": "Parties-Answer",
				"termination for convenience": "Termination For Convenience-Answer",
				"exclusivity": "Exclusivity-Answer",
				"revenue": "Revenue/Profit Sharing-Answer",
				"profit sharing": "Revenue/Profit Sharing-Answer",
			}
			selected_label = None
			for key, lbl in label_map.items():
				if key in _q_lower:
					selected_label = lbl
					break

			def _extract_answer(text: str, label: str) -> str | None:
				if not text:
					return None
				prefix = f"{label}:"
				lines = str(text).splitlines()
				for idx, line in enumerate(lines):
					if line.strip().lower().startswith(prefix.lower()):
						val = line.split(":", 1)[1].strip()
						return val.strip().strip("[]\"'")
				return None

			def _support_from_lines(text: str, label: str) -> list[str]:
				lines = str(text).splitlines()
				prefix = f"{label}:"
				out = []
				for i, line in enumerate(lines):
					if line.strip().lower().startswith(prefix.lower()):
						for j in range(i+1, min(i+4, len(lines))):
							candidate = lines[j].strip()
							if candidate:
								out.append(candidate[:200])
								if len(out) >= 2:
									return out
				return out

			if selected_label:
				cand_df = vec.search(
					selected_label,
					limit=5,
					metadata_filter={"filename": uploaded_file.name},
				)
				series = cand_df.get("content") if "content" in cand_df.columns else cand_df.get("contents")
				if series is not None:
					for txt in series.tolist():
						ans = _extract_answer(txt, selected_label)
						if ans:
							st.subheader("Answer")
							st.write(ans)
							snips = _support_from_lines(txt, selected_label)
							if snips:
								st.caption("Supporting context:")
								for s in snips:
									st.write(f"- {s}")
							pdf_file = generate_pdf_with_features(ans, uploaded_file.name, "Q&A Report")
							reports_dir = Path(__file__).resolve().parent.parent / "reports"
							reports_dir.mkdir(parents=True, exist_ok=True)
							filename = f"{uploaded_file.name.rsplit('.', 1)[0]}_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
							(out_path := reports_dir / filename).write_bytes(pdf_file.getbuffer())
							st.download_button(
								label="Download Answer as PDF",
								data=pdf_file,
								file_name=filename,
								mime="application/pdf",
							)
							st.stop()
				# Label detected but value not found
				st.subheader("Answer")
				st.write("Cannot determine from the provided text.")
				st.stop()

		# LLM fallback disabled to avoid payload/quota; relying on structured answers only

# Footer
st.markdown("---")
st.markdown("Powered by **Streamlit**, **PyPDF2**, **ReportLab**, and **Google Gemini**")

# Diagnostics: show which key clauses are present in retrieved chunks for this file
with st.expander("Document diagnostics (clause presence)"):
	if uploaded_file and st.session_state.pdf_content.strip():
		try:
			clause_queries = {
				"Indemnity": ["indemnity", "indemnification"],
				"Limitation of Liability": ["limitation of liability", "liability cap"],
				"Confidentiality": ["confidential", "non-disclosure"],
				"Governing Law": ["governing law", "jurisdiction"],
				"Termination": ["termination", "terminate"],
				"Warranties": ["warranty", "warranties"],
				"Intellectual Property": ["intellectual property", "IP ownership", "license"],
				"Service Levels": ["service level", "SLA"],
			}
			for clause, terms in clause_queries.items():
				# Use the first term as a retrieval seed, filter by filename
				seed = terms[0]
				res_df = vec.search(
					seed,
					limit=3,
					metadata_filter={"filename": uploaded_file.name},
				)
				contents_series = res_df.get("content") if "content" in res_df.columns else res_df.get("contents")
				rows = contents_series.tolist() if contents_series is not None else []
				found = False
				for row_text in rows:
					row_lower = str(row_text).lower()
					if any(term.lower() in row_lower for term in terms):
						found = True
						break
				st.write(f"- {clause}: {'✅ Found' if found else '❌ Not found in retrieved chunks'}")
		except Exception as e:
			st.info(f"Diagnostics unavailable: {e}")
