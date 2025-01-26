from fastapi import FastAPI
from src.controllers.controllers import router as pdf_router

app = FastAPI(
    title="Compliance Checker API",
    description="PDF Upload and Processing API",
    version="0.1.0"
)

app.include_router(pdf_router, prefix="/api/pdf", tags=["PDF"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Compliance Checker API"}
