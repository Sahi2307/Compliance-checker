from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.services.similarity_search import SimilaritySearchEngine
from app.services.compliance_evaluator import ComplianceEvaluator
from app.services.llm_compliance_analyzer import LLMComplianceAnalyzer
from app.services.report_generator import EnhancedComplianceReportGenerator
from app.managers.ingestion_manager import save_file, extract_text_from_pdf
from app.managers.embedding_manager import generate_and_store_embeddings
import os
import pandas as pd

router = APIRouter()

@router.post("/analyze/")  
async def advanced_document_analysis(file: UploadFile = File(...), query: str = "Comprehensive document analysis"):
    """
    Advanced document analysis with compliance evaluation
    """
    try:
        # Save and process file
        file_location = save_file(file, UPLOAD_DIR)
        generate_and_store_embeddings(file_location)

        # Perform similarity search
        search_engine = SimilaritySearchEngine()
        search_results = search_engine.search(query)

        # Compliance evaluation
        compliance_evaluator = ComplianceEvaluator()
        compliance_result = compliance_evaluator.evaluate_document(search_results)

        # LLM-powered insights
        llm_analyzer = LLMComplianceAnalyzer()
        document_text = " ".join(search_results['content'])
        llm_insights = llm_analyzer.analyze_document(document_text)

        # Generate enhanced report
        report_generator = EnhancedComplianceReportGenerator()
        report_filename = f"compliance_report_{file.filename.split('.')[0]}.pdf"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        
        report_generator.generate_report(
            query=query,
            search_results=search_results,
            compliance_result=compliance_result,
            llm_insights=llm_insights,
            output_path=report_path
        )

        return JSONResponse(content={
            "message": "Advanced document analysis completed",
            "compliance_score": compliance_result.overall_score,
            "risk_level": compliance_result.risk_level,
            "report_path": report_filename
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
