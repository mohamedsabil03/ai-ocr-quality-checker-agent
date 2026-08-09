from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class OCRCheckRequest(BaseModel):
    ocr_text: str = Field(..., description="The raw OCR text output to evaluate", json_schema_extra={"example": "INVOICE #1024\nDate: 2026-08-01\nTotal: $450.00"})
    ground_truth_text: Optional[str] = Field(None, description="Reference ground truth text if available for comparison", json_schema_extra={"example": "INVOICE #1024\nDate: 2026-08-01\nTotal: $450.00"})
    mandatory_fields: Optional[List[str]] = Field(
        default_factory=lambda: ["Invoice Number", "Date", "Total Amount"],
        description="List of required key fields expected in the document",
        json_schema_extra={"example": ["Invoice Number", "Date", "Total Amount"]}
    )
    target_language: Optional[str] = Field("en", description="Expected language ISO code (e.g. 'en', 'es', 'de', 'fr')")
    model_name: Optional[str] = Field("qwen3", description="SLM model to use: 'qwen3' or 'phi4'")
    enable_tools: Optional[List[str]] = Field(
        default_factory=lambda: ["missing_text", "mandatory_fields", "language_check"],
        description="Active evaluation tools",
        json_schema_extra={"example": ["missing_text", "mandatory_fields", "language_check"]}
    )


class MissingTextResult(BaseModel):
    similarity_score: float = Field(..., description="String similarity score between OCR and Ground Truth (0.0 to 1.0)")
    missing_snippets: List[str] = Field(default_factory=list, description="Text segments present in ground truth but missing in OCR")
    missing_count: int = Field(0, description="Total count of missing text segments")
    details: str = Field("", description="Detailed explanation of omissions")


class MandatoryFieldsResult(BaseModel):
    found_fields: List[str] = Field(default_factory=list, description="Fields identified in OCR text")
    missing_fields: List[str] = Field(default_factory=list, description="Fields missing from OCR text")
    field_presence_ratio: float = Field(..., description="Ratio of present mandatory fields (0.0 to 1.0)")
    field_details: Dict[str, bool] = Field(default_factory=dict, description="Presence boolean for each field")


class LanguageCheckResult(BaseModel):
    detected_language: str = Field(..., description="Detected language code")
    expected_language: str = Field(..., description="Target expected language code")
    is_match: bool = Field(..., description="Whether detected language matches target language")
    confidence: float = Field(..., description="Language detection confidence score (0.0 to 1.0)")
    lang_score: float = Field(..., description="Language quality and fluency score (0.0 to 1.0)")
    details: str = Field("", description="Analysis of language quality and character degradation")


class ToolResults(BaseModel):
    missing_text: MissingTextResult
    mandatory_fields: MandatoryFieldsResult
    language_check: LanguageCheckResult


class QualityScoreBreakdown(BaseModel):
    overall_score: float = Field(..., description="Aggregated Quality Score from 0 to 100")
    completeness_score: float = Field(..., description="Text completeness sub-score (0 to 100)")
    field_accuracy_score: float = Field(..., description="Mandatory field accuracy sub-score (0 to 100)")
    language_score: float = Field(..., description="Language validity sub-score (0 to 100)")
    grade: str = Field(..., description="Quality grade: EXCELLENT, GOOD, NEEDS_REVIEW, POOR")
    summary: str = Field(..., description="High-level summary of document quality")
    recommendations: List[str] = Field(default_factory=list, description="Actionable suggestions for quality improvement")


class OCRCheckResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique evaluation request ID")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp")
    model_used: str = Field(..., description="Small Language Model or Engine used for reasoning")
    score: QualityScoreBreakdown
    tool_results: ToolResults
    agent_reasoning: str = Field(..., description="Detailed agent chain-of-thought analysis")