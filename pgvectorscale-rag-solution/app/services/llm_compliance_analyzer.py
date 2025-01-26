from app.services.llm_factory import LLMFactory
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

class LegalComplianceInsight(BaseModel):
    section: str = "Document Overview"
    potential_risks: List[str] = []
    improvement_suggestions: List[str] = []
    confidence_score: float = 0.7

class LLMComplianceAnalyzer:
    def __init__(self, llm_provider: str = "huggingface"):
        """
        Initialize LLM-powered compliance analyzer
        
        Args:
            llm_provider: LLM provider to use for analysis
        """
        try:
            self.llm_factory = LLMFactory(llm_provider)
        except Exception as e:
            logging.error(f"Failed to initialize LLM Factory: {e}")
            raise

    def analyze_document(self, document_text: str) -> List[LegalComplianceInsight]:
        """
        Use LLM to generate detailed compliance insights
        
        Args:
            document_text: Full text of the document to analyze
        
        Returns:
            List of compliance insights
        """
        # Truncate document text if it's too long
        document_text = document_text[:2000]  # Limit to first 2000 characters

        messages = [
            {
                "role": "system",
                "content": """You are a legal compliance expert. Analyze the following document 
                and provide insights on potential legal risks, compliance issues, 
                and improvement suggestions."""
            },
            {
                "role": "user",
                "content": f"Analyze this document for legal compliance: {document_text}"
            }
        ]

        try:
            # Use LLM to generate insights
            response = self.llm_factory.create_completion(
                response_model=List[LegalComplianceInsight],
                messages=messages
            )
            
            # If response doesn't return a list, create a default insight
            if not response.data:
                return [LegalComplianceInsight(
                    potential_risks=["Unable to generate detailed insights"],
                    improvement_suggestions=["Manual review recommended"]
                )]
            
            return response.data
        except Exception as e:
            logging.error(f"Compliance analysis error: {e}")
            return [LegalComplianceInsight(
                potential_risks=[f"Analysis failed: {str(e)}"],
                improvement_suggestions=["Manual review required"]
            )]