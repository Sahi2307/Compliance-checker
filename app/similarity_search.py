from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------

#relevant_question = "What is the most frequently occuring exact law in the dataset?"
relevant_question = "Explain the discrepancy of CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf"
results = vec.search(relevant_question, limit=3)
print(results.columns)


response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

"""irrelevant_question = "Explain the discrepancy of CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf"

results = vec.search(irrelevant_question, limit=3)

response = Synthesizer.generate_response(question=irrelevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

 """


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

# The string to be printed to the PDF
response_answer = response.answer

# Function to convert the string to a list of Paragraph objects
def convert_to_paragraphs(text):
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    bold_style = styles['Heading1']

    lines = text.split('\n')
    paragraphs = []

    for line in lines:
        if line.startswith('**') and line.endswith('**'):
            # Bold text
            paragraphs.append(Paragraph(line.strip('**'), bold_style))
        elif line.startswith('- '):
            # Bullet points
            paragraphs.append(Paragraph(line, normal_style))
        else:
            # Normal text
            paragraphs.append(Paragraph(line, normal_style))

    return paragraphs

# Create a PDF document
pdf_filename = "suitability_report.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter)

# Convert the string to a list of Paragraph objects
paragraphs = convert_to_paragraphs(response_answer)

# Build the PDF
doc.build(paragraphs)

print(f"PDF generated: {pdf_filename}") 








