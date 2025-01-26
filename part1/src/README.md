# AI-Powered Regulatory Compliance Checker for Contracts  

This project is an AI-driven tool designed to analyze contracts for compliance with legal standards. Users can upload contract PDFs, and the system identifies potential compliance issues and provides detailed feedback.  

## Features  
- **PDF File Ingestion**: API for uploading and validating PDF files.  
- **Text Extraction**: Robust text extraction with edge-case handling.  
- **Data Chunking**: Python function to split extracted text into smaller, overlapping chunks for processing.  
- **Text Embeddings**: Generates embeddings for text chunks using a large language model (LLM) or pre-trained model.  
- **Vector Store**:  
  - **Local**: Creates and manages FAISS indexes for embeddings.  
  - **Server**: Integration with hosted vector databases like Pinecone.  
- **Compliance Analysis**: Compares contract contents with legal standards to detect compliance issues and provides actionable insights.  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Sahi2307/Compliance-checker.git 
   cd contract-compliance-checker  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt.txt  
   ```  

## Usage  
1. Start the API server for PDF uploads:  
   ```bash  
   python api_server.py  
   ```  
2. Process a PDF:  
   - Upload a PDF through the API endpoint.  
   - Extracted text is chunked, embedded, and stored in a vector database.  
3. Run the compliance checker:  
   ```bash  
   python compliance_checker.py  
   ```  
4. View results:  
   - The system will flag any compliance issues and generate a detailed message based on the uploaded contract.  

## Technologies Used  
- **Python**  
- **LangChain**  
- **FAISS**  
- **Pinecone**  
- **Large Language Models (LLMs)**  

## Future Enhancements  
- Support for additional document formats.  
- Enhanced compliance rules tailored to specific legal frameworks.  
- Integration with external legal databases.  
  

## Contributing  
Contributions are welcome! Please open an issue or submit a pull request.  

