import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ComplianceEvaluationResult(BaseModel):
    overall_score: float = Field(ge=0, le=100)
    risk_level: str = Field(default="Unknown")
    critical_findings: List[str] = []
    recommendations: List[str] = []

class ComplianceEvaluator:
    def __init__(self, rules_config: Optional[Dict] = None):
        """
        Initialize compliance evaluator with optional custom rules
        
        Args:
            rules_config: Dictionary of custom compliance rules
        """
        self.default_rules = {
            "mandatory_clauses": [
                "Governing Law", 
                "Termination Conditions", 
                "Confidentiality"
            ],
            "risk_thresholds": {
                "high": 70,
                "medium": 50,
                "low": 30
            }
        }
        self.rules = rules_config or self.default_rules

    def evaluate_document(self, document_data: pd.DataFrame) -> ComplianceEvaluationResult:
        """
        Comprehensive compliance evaluation of a document
        
        Args:
            document_data: DataFrame containing document details
        
        Returns:
            Structured compliance evaluation result
        """
        score = 0
        critical_findings = []
        recommendations = []

        # Check mandatory clauses
        for clause in self.rules['mandatory_clauses']:
            if clause not in document_data.columns or document_data[clause].isnull().all():
                critical_findings.append(f"Missing mandatory clause: {clause}")
                score -= 10

        # Basic scoring logic
        total_columns = len(document_data.columns)
        filled_columns = document_data.notna().sum().sum()
        completeness_score = (filled_columns / total_columns) * 100
        score += min(completeness_score, 50)  # Cap completeness contribution

        # Risk assessment
        risk_level = "Low"
        if score <= self.rules['risk_thresholds']['high']:
            risk_level = "High"
        elif score <= self.rules['risk_thresholds']['medium']:
            risk_level = "Medium"

        # Generate recommendations
        if critical_findings:
            recommendations = [
                "Review and include all mandatory clauses",
                "Consult legal counsel for comprehensive document review"
            ]

        return ComplianceEvaluationResult(
            overall_score=max(0, min(score, 100)),
            risk_level=risk_level,
            critical_findings=critical_findings,
            recommendations=recommendations
        )
