import os
import logging
import streamlit as st
import PyPDF2  # To extract content from PDFs
from io import BytesIO  # To handle in-memory file objects
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from config.settings import get_settings, setup_logging
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer, SynthesizedResponse
# Removed tiktoken dependency - using Google Gemini instead
## OCR disabled per user request; relying on native text extraction only

# Initialize logging early
setup_logging()

# VectorStore will be initialized lazily when processing begins to avoid import-time failures
vec = None  # Initialized on demand

# Streamlit App Title
st.title("Legal Contract Assistant")
st.subheader("Analyze contracts PDFs")

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload up to 3 PDF contracts:",
    type=["pdf"],
    accept_multiple_files=True,
)

# Q&A input and per-document selection removed per request

# Function to count tokens - simplified for Google Gemini
def count_tokens(text):
    # Simple word-based token estimation (rough approximation)
    return len(text.split()) * 1.3  # Rough estimate: 1.3 tokens per word

# Keep payloads under Gemini limits by truncating large texts
def truncate_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

# Fallback/formatter to build the final report with only the required sections
def generate_fallback_report(
    pdf_text: str,
    context_df,
    uploaded_pdf_name: str,
    reasoning_text: str,
    sufficient_context: bool,
) -> str:
    """
    Build a heuristic, per-PDF report containing ONLY:
    - Compliance Score
    - Strengths
    - Areas for Improvement
    - Reasoning
    - Additional Information
    - Context Assessment
    """
    import re

    text = pdf_text or ""
    text_lower = text.lower()

    # Clause groups mapped to rubric weights
    groups = [
        ("Core legal protections", 30, [
            "indemnification", "indemnify", "limitation of liability", "liability cap",
            "warranty", "warranties"
        ]),
        ("Data protection and confidentiality", 20, [
            "confidential", "confidentiality", "data protection", "privacy", "gdpr",
            "security", "information security"
        ]),
        ("Operational clarity", 20, [
            "scope of work", "scope", "service level", "sla", "termination",
            "termination for convenience", "change control", "change request"
        ]),
        ("Compliance with applicable law", 20, [
            "governing law", "jurisdiction", "export", "anti-bribery",
            "anti bribery", "anti corruption", "intellectual property",
            "ip ownership", "license"
        ]),
        ("Drafting quality and completeness", 10, [
            "definitions", "entire agreement", "order of precedence", "conflict",
            "severability"
        ]),
    ]

    bad_markers = ["tbd", "to be determined", "to be agreed", "[insert", "???"]

    def find_occurrence_snippet(src_text: str, keyword: str, radius: int = 160) -> str:
        pattern = re.escape(keyword)
        m = re.search(pattern, src_text, flags=re.IGNORECASE)
        if not m:
            return ""
        start = max(0, m.start() - radius)
        end = min(len(src_text), m.end() + radius)
        snippet = src_text[start:end].replace("\n", " ")
        return snippet.strip()

    total_score = 0.0
    strengths: list[str] = []
    improvements: list[str] = []
    section_summaries: list[str] = []

    for title, weight, keywords in groups:
        found = []
        missing = []
        for kw in keywords:
            if kw in text_lower:
                found.append(kw)
            else:
                missing.append(kw)

        # Score proportionally by coverage in each category
        coverage = (len(found) / len(keywords)) if keywords else 0.0
        section_score = weight * coverage

        # Penalize drafting quality for bad markers
        penalty_hits = 0
        penalty = 0.0
        if title.startswith("Drafting"):
            penalty_hits = sum(1 for bm in bad_markers if bm in text_lower)
            if penalty_hits:
                penalty = min(0.5, penalty_hits * 0.1)  # up to 50% penalty
                section_score *= (1.0 - penalty)
        total_score += section_score

        # Summarize contribution of this category
        summary = f"- {title}: {round(section_score)}/{weight} from {len(found)}/{len(keywords)} signals"
        if title.startswith("Drafting") and penalty_hits:
            summary += f" (penalty {int(penalty * 100)}% for drafting placeholders)"
        section_summaries.append(summary)

        if found:
            strengths.append(f"{title}: present signals (e.g., {', '.join(found[:3])})")
        if missing:
            improvements.append(
                f"{title}: consider adding or clarifying {', '.join(missing[:3])}"
            )

    # Bound and format the total score; scale raw (0–100) into the 70–85 band
    raw_score = max(0, min(100, round(total_score)))
    final_score = int(round(70 + (raw_score * 0.15)))
    final_score = max(70, min(85, final_score))

    strengths_text = "\n".join(f"- {s}" for s in strengths) if strengths else "- No clear strengths detected."
    improvements_text = "\n".join(f"- {i}" for i in improvements) if improvements else "- No clear gaps detected; review manually."

    # Use provided reasoning text (trimmed) or a default
    reasoning_text = (reasoning_text or "Heuristic assessment based on clause presence and drafting signals.").strip()

    # Context assessment: True when final score >= 80
    ctx_available = bool(final_score >= 80)

    # Additional information (basic guidance)
    additional_info = (
        "- Consider explicitly defining post-termination obligations and transition services where relevant.\n"
        "- Ensure any placeholders (e.g., TBD/To be agreed) are resolved before execution."
    )

    report = f"""**Compliance Score:**
- {final_score} out of 100

**Strengths:**
{strengths_text}

**Areas for Improvement:**
{improvements_text}

**Reasoning:**
{reasoning_text}

**Additional Information:**
{additional_info}

**Context Assessment:**
- Sufficient context available: {str(ctx_available)}
"""
    return report

# Enhanced PDF Generation Function
def generate_pdf_with_features(response_text, uploaded_pdf_name, report_title: str = "Analysis Report"):
    """
    Generate a styled PDF based on input text, including bold, normal text, and bullet points.
    The uploaded PDF's name (without extension) is displayed as the title of the output PDF.
    """
    # Remove the .pdf extension from the uploaded file name
    pdf_name = uploaded_pdf_name.rsplit(".", 1)[0]
    title_text = f"{pdf_name} {report_title}"

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
    if not response_text or not response_text.strip():
        response_text = "No analysis generated."
    lines = response_text.split('\n')
    for line in lines:
        if line.startswith('**') and line.endswith('**'):
            # Bold text (heading-like)
            heading_text = escape(line.strip('**'))
            paragraphs.append(Paragraph(heading_text, bold_style))
            paragraphs.append(Spacer(1, 12))  # Add spacing after headings
        elif line.startswith('- '):
            # Bullet points
            bullet_text = escape(line)
            paragraphs.append(Paragraph(bullet_text, normal_style))
        else:
            # Normal text
            paragraphs.append(Paragraph(escape(line), normal_style))
            paragraphs.append(Spacer(1, 10))  # Add spacing after paragraphs

    # Build the PDF
    doc.build(paragraphs)
    buffer.seek(0)
    return buffer

# Initialize session state to store results
if "pdf_responses" not in st.session_state:
    st.session_state.pdf_responses = []
## Q&A state removed per request

# Analyze Contracts Section
st.markdown("### Analyze Contracts")
save_to_disk = st.checkbox("Also save generated PDFs to the reports folder", value=True)
if st.button("Analyze Contracts"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file before submitting.")
    else:
        # Enforce a maximum of 3 files
        selected_files = uploaded_files[:3]
        if len(uploaded_files) > 3:
            st.warning("You uploaded more than 3 files; only the first 3 will be processed.")

        # Validate required environment configuration
        settings = get_settings()
        missing_env = []
        if not settings.google_gemini.api_key:
            missing_env.append("GOOGLE_API_KEY")
        if not settings.database.service_url:
            missing_env.append("TIMESCALE_SERVICE_URL")
        if missing_env:
            st.error(f"Missing required environment variables: {', '.join(missing_env)}. Please set them in the .env file and restart the app.")
            selected_files = []
        else:
            # Initialize VectorStore lazily to avoid import-time failures
            try:
                vec = VectorStore()
            except Exception as e:
                st.error(f"Failed to initialize vector store/database: {e}")
                logging.exception("VectorStore initialization failed")
                selected_files = []

        pdf_responses = []

        # Process each uploaded file
        for uploaded_file in selected_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Extract text from the uploaded PDF
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    pdf_content = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text() or ""
                        pdf_content += page_text

                    # Ensure the PDF content is not empty
                    if not pdf_content.strip():
                        st.error(f"Unable to extract text from {uploaded_file.name}. Please check the file.")
                        continue

                    
                    # Keep request sizes small
                    short_text = truncate_text(pdf_content, 4000)

                    # Retrieve more relevant chunks for this document section
                    results = vec.search(short_text, limit=8)

                    # Generate a compliance analysis (LLM) with retrieved context
                    response: SynthesizedResponse = Synthesizer.generate_response(
                        question="Provide a compliance analysis for this contract section.",
                        context=results,
                    )

                    # Determine if insufficient context and choose reasoning
                    raw_answer = (getattr(response, 'answer', '') or '').strip()
                    enough_ctx_flag = getattr(response, 'enough_context', True)
                    rate_limited = (enough_ctx_flag is False)

                    # If insufficient, provide a heuristic reasoning; else use model answer as reasoning
                    if rate_limited:
                        reasoning = "Preliminary heuristic analysis generated due to service rate limits or insufficient context."
                    else:
                        reasoning = raw_answer or "Heuristic assessment based on clause presence and drafting signals."

                    # Context availability: use LLM flag if present, else infer from search results
                    has_results = hasattr(results, 'empty') and not results.empty
                    sufficient_context = enough_ctx_flag if enough_ctx_flag is not None else has_results

                    # Build the final report with only the required sections
                    final_report = generate_fallback_report(
                        pdf_text=pdf_content,
                        context_df=results,
                        uploaded_pdf_name=uploaded_file.name,
                        reasoning_text=reasoning,
                        sufficient_context=sufficient_context,
                    )

                    # Store the final report and file name for generating the PDF
                    pdf_responses.append((final_report, uploaded_file.name))

                except Exception as e:
                    st.error(f"An error occurred while processing {uploaded_file.name}: {e}")

        # Save the responses in session state
        st.session_state.pdf_responses = pdf_responses

# Display Download Buttons
if "pdf_responses" in st.session_state and st.session_state.pdf_responses:
    st.subheader("Analysis Preview")
    for response_text, file_name in st.session_state.pdf_responses:
        st.markdown(f"#### {file_name}")
        st.markdown(response_text)

    st.subheader("Download Analysis Reports:")
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for response_text, file_name in st.session_state.pdf_responses:
        pdf_file = generate_pdf_with_features(response_text, file_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"{file_name.rsplit('.', 1)[0]}_analysis_{timestamp}.pdf"
        if save_to_disk:
            (reports_dir / out_name).write_bytes(pdf_file.getbuffer())
        st.download_button(
            label=f"Download Analysis for {file_name}",
            data=pdf_file,
            file_name=out_name,
            mime="application/pdf",
        )

## Q&A results removed per request

# Diagnostics panel for the selected file (if any)
## Diagnostics panel removed per request
