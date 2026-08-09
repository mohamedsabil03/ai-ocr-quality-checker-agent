import pytest
from app.tools.missing_text import run_missing_text_tool
from app.tools.mandatory_fields import run_mandatory_fields_tool
from app.tools.language_check import run_language_check_tool
from app.scoring.quality_score import calculate_quality_score


def test_missing_text_tool_with_ground_truth():
    gt = "INVOICE #101\nDate: 2026-08-01\nTotal: $100.00"
    ocr = "INVOICE #101\nTotal: $100.00"
    
    result = run_missing_text_tool(ocr_text=ocr, ground_truth_text=gt)
    
    assert result.similarity_score < 1.0
    assert result.missing_count >= 1
    assert any("Date" in s for s in result.missing_snippets)


def test_mandatory_fields_tool_all_found():
    ocr = "INVOICE #INV-8812\nDate: 2026-08-01\nTotal Amount: $500.00"
    fields = ["Invoice Number", "Date", "Total Amount"]
    
    result = run_mandatory_fields_tool(ocr_text=ocr, mandatory_fields=fields)
    
    assert result.field_presence_ratio == 1.0
    assert len(result.found_fields) == 3
    assert len(result.missing_fields) == 0


def test_mandatory_fields_tool_missing_field():
    ocr = "INVOICE #INV-8812\nTotal Amount: $500.00"
    fields = ["Invoice Number", "Date", "Total Amount"]
    
    result = run_mandatory_fields_tool(ocr_text=ocr, mandatory_fields=fields)
    
    assert result.field_presence_ratio < 1.0
    assert "Date" in result.missing_fields


def test_language_check_tool_english():
    ocr = "This is a clean invoice document written in English."
    result = run_language_check_tool(ocr_text=ocr, target_language="en")
    
    assert result.detected_language == "en"
    assert result.is_match is True
    assert result.lang_score > 0.7


def test_scoring_engine_perfect_score():
    missing_res = run_missing_text_tool("Full Text", "Full Text")
    field_res = run_mandatory_fields_tool("INVOICE #100\nDate: 2026-08-01\nTotal Amount: $50", ["Invoice Number", "Date", "Total Amount"])
    lang_res = run_language_check_tool("Full English text invoice", "en")
    
    score_breakdown = calculate_quality_score(missing_res, field_res, lang_res)
    
    assert score_breakdown.overall_score >= 90.0
    assert score_breakdown.grade == "EXCELLENT"


def test_react_agent_tool_calling_loop():
    from app.agent.ocr_agent import OCRAgent
    from app.schemas import OCRCheckRequest

    agent = OCRAgent(default_model="qwen3")
    req = OCRCheckRequest(
        ocr_text="INVOICE #INV-900\nDate: 2026-08-01\nTotal Amount: $200.00",
        mandatory_fields=["Invoice Number", "Date", "Total Amount"],
        target_language="en",
        model_name="qwen3"
    )
    res = agent.evaluate(req)

    assert res.model_used == "qwen3"
    assert "Autonomous LLM Tool-Calling Agent Execution Log" in res.agent_reasoning
    assert "missing_text_tool" in res.agent_reasoning
    assert "mandatory_fields_tool" in res.agent_reasoning
    assert "language_check_tool" in res.agent_reasoning
    assert res.score.overall_score >= 80.0

