from typing import List  # Import List from typing
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pandas as pd
from app.services.compliance_evaluator import ComplianceEvaluationResult
from app.services.llm_compliance_analyzer import LegalComplianceInsight

class EnhancedComplianceReportGenerator:
    def generate_report(
        self, 
        query: str, 
        search_results: pd.DataFrame, 
        compliance_result: ComplianceEvaluationResult,
        llm_insights: List[LegalComplianceInsight],
        output_path: str
    ):
        """
        Generate a comprehensive PDF compliance report
        
        Args:
            query: Original search query
            search_results: DataFrame of search results
            compliance_result: Compliance evaluation result
            llm_insights: LLM-generated compliance insights
            output_path: Path to save the report
        """
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []

        # Title and Overview
        story.append(Paragraph("Comprehensive Compliance Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Query: {query}", styles['Heading2']))
        
        # Compliance Score Section
        story.append(Paragraph("Compliance Assessment", styles['Heading2']))
        story.append(Paragraph(f"Overall Score: {compliance_result.overall_score:.2f}/100", styles['Normal']))
        story.append(Paragraph(f"Risk Level: {compliance_result.risk_level}", styles['Normal']))
        
        # Critical Findings
        if compliance_result.critical_findings:
            story.append(Paragraph("Critical Findings:", styles['Heading3']))
            for finding in compliance_result.critical_findings:
                story.append(Paragraph(f"• {finding}", styles['BodyText']))
        
        # Recommendations
        if compliance_result.recommendations:
            story.append(Paragraph("Recommendations:", styles['Heading3']))
            for rec in compliance_result.recommendations:
                story.append(Paragraph(f"• {rec}", styles['BodyText']))
        
        # LLM Insights
        if llm_insights:
            story.append(Paragraph("Advanced Legal Insights", styles['Heading2']))
            for insight in llm_insights:
                story.append(Paragraph(f"Section: {insight.section}", styles['Heading3']))
                story.append(Paragraph(f"Confidence: {insight.confidence_score:.2%}", styles['Normal']))
                
                story.append(Paragraph("Potential Risks:", styles['Heading4']))
                for risk in insight.potential_risks:
                    story.append(Paragraph(f"• {risk}", styles['BodyText']))
                
                story.append(Paragraph("Improvement Suggestions:", styles['Heading4']))
                for suggestion in insight.improvement_suggestions:
                    story.append(Paragraph(f"• {suggestion}", styles['BodyText']))

        doc.build(story)
        return output_path
