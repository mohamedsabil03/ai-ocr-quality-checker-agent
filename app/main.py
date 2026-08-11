import os
import json
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import OCRCheckRequest, OCRCheckResponse
from app.agent.ocr_agent import OCRAgent
from app.agent.model_loader import model_loader

app = FastAPI(
    title="AI OCR Quality Checker API",
    
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "OCR Evaluation", "description": "Core OCR quality assessment endpoints"},
        {"name": "System & Models", "description": "Service status and SLM model management"}
    ]
)

# Initialize OCR Quality Agent
agent = OCRAgent(default_model="qwen3")

# Static files setup
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serves the interactive UI dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>AI OCR Quality Checker API is running. UI dashboard not found.</h2>")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Favicon endpoint to eliminate browser 404 logs."""
    return Response(status_code=204)


@app.get("/sample/invoice", tags=["OCR Evaluation"], summary="Get Sample Invoice Payload", description="Retrieve a sample invoice dataset item for OCR testing and benchmarking.")
async def get_sample_invoice(sample_id: str = None):
    """Returns sample invoice test data from dataset."""
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "test.json")
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if sample_id:
                sample = next((item for item in data if item.get("id") == sample_id), None)
                if sample:
                    return sample
                raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
            return data[0] if data else {}
        except json.JSONDecodeError:
            pass

    return {
        "id": "tc_001_perfect_invoice",
        "name": "Perfect Invoice Extraction",
        "target_language": "en",
        "mandatory_fields": ["Invoice Number", "Date", "Total Amount", "Vendor"],
        "ocr_text": "ACME LOGISTICS INC.\n100 Tech Way, Suite 400\nINVOICE #INV-2026-8891\nDate: 2026-08-01\nVendor: ACME Logistics\n\nDescription: Cloud Infrastructure Consulting Services\nSubtotal: $4,500.00\nTax (10%): $450.00\nTotal Amount: $4,950.00\n\nThank you for your business!",
        "ground_truth_text": "ACME LOGISTICS INC.\n100 Tech Way, Suite 400\nINVOICE #INV-2026-8891\nDate: 2026-08-01\nVendor: ACME Logistics\n\nDescription: Cloud Infrastructure Consulting Services\nSubtotal: $4,500.00\nTax (10%): $450.00\nTotal Amount: $4,950.00\n\nThank you for your business!"
    }


@app.get("/health", tags=["System & Models"], summary="Service Health & Status", description="Check system operational status and loaded SLM models.")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "AI OCR Quality Checker API",
        "version": "1.0.0",
        "models": model_loader.get_available_models()
    }


@app.get("/models", tags=["System & Models"], summary="List SLM Models", description="Retrieve available local Small Language Model weights (Qwen3-4B-Instruct).")
async def list_models():
    """Returns available local SLM models and load status."""
    return model_loader.get_available_models()


@app.post("/models/load", tags=["System & Models"], summary="Explicitly Load SLM Model", description="Triggers loading of Qwen3 model weights into memory.")
async def load_model_endpoint(model_name: str = "qwen3"):
    """Loads specified SLM model into memory."""
    try:
        success = model_loader.load_model(model_name)
        return {
            "success": success,
            "model_name": model_name,
            "status": model_loader.get_available_models()
        }
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))


@app.post(
    "/ocr/check",
    response_model=OCRCheckResponse,
    tags=["OCR Evaluation"],
    summary="Evaluate OCR Text Quality",
    description="Full agentic evaluation pipeline: executes missing text, mandatory fields, and language diagnostic tools, calculates 0–100 quality score, and synthesizes SLM agent reasoning."
)
async def check_ocr_quality(request: OCRCheckRequest):
    """
    Evaluates OCR quality using diagnostic tools (Missing Text, Mandatory Fields, Language Check),
    scoring engine, and Small Language Model reasoning.
    """
    if not request.ocr_text:
        raise HTTPException(status_code=400, detail="Field 'ocr_text' must not be empty.")
    
    try:
        response = agent.evaluate(request)
        return response
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR evaluation error: {str(e)}")