# AI-Powered Regulatory Compliance Checker for Contracts

## Overview
Advanced AI-powered tool for legal document analysis using Falcon LLM and BERT, focusing on regulatory compliance and intelligent document insights.

## Technology Stack
- Python
- Streamlit
- Hugging Face (Falcon-7b, BERT)
- PostgreSQL
- LangChain
- scikit-learn

## Key Features
- PDF document analysis
- Regulatory compliance scoring
- Intelligent legal component extraction
- Database-backed document comparison
- Comprehensive report generation

## Prerequisites
- Python 3.8+
- PostgreSQL
- Hugging Face API Key

## Installation

1. Clone Repository
```bash
git clone https://github.com/yourusername/ai-regulatory-compliance-checker.git
cd ai-regulatory-compliance-checker
```

2. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Database Configuration
- Create PostgreSQL database
- Update `config/settings.py`

5. Run Application
```bash
streamlit run main.py
```

## Configuration
- Set Hugging Face API token in `.env`
- Customize database settings

## Usage
1. Upload PDF contract
2. System analyzes document
3. View compliance report
4. Download detailed analysis

## Contributing
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## License
MIT License
