from typing import List, Tuple
from app.schemas import (
    MissingTextResult,
    MandatoryFieldsResult,
    LanguageCheckResult,
    QualityScoreBreakdown
)


def calculate_quality_score(
    missing_res: MissingTextResult,
    field_res: MandatoryFieldsResult,
    lang_res: LanguageCheckResult
) -> QualityScoreBreakdown:
    """
    Computes a overall OCR Quality Score (0-100) based on weighted tool metrics.
    """
    # 1. Sub-scores (scaled to 0-100)
    completeness_score = round(missing_res.similarity_score * 100.0, 1)
    field_accuracy_score = round(field_res.field_presence_ratio * 100.0, 1)
    language_score = round(lang_res.lang_score * 100.0, 1)

    # 2. Weighted Overall Score (40% Completeness, 40% Fields, 20% Language)
    raw_overall = (
        0.40 * completeness_score +
        0.40 * field_accuracy_score +
        0.20 * language_score
    )
    overall_score = round(max(0.0, min(100.0, raw_overall)), 1)

    # 3. Grade determination
    if overall_score >= 90.0:
        grade = "EXCELLENT"
    elif overall_score >= 75.0:
        grade = "GOOD"
    elif overall_score >= 50.0:
        grade = "NEEDS_REVIEW"
    else:
        grade = "POOR"

    # 4. Generate recommendations & summary
    recommendations: List[str] = []
    
    if field_res.missing_fields:
        missing_str = ", ".join(field_res.missing_fields)
        recommendations.append(f"Missing mandatory fields detected: [{missing_str}]. Verify document crop or re-scan.")
        
    if missing_res.missing_count > 0:
        recommendations.append(f"Detected {missing_res.missing_count} omitted text block(s). Check OCR image thresholding/resolution.")
        
    if not lang_res.is_match:
        recommendations.append(f"Document language ('{lang_res.detected_language}') differs from target ('{lang_res.expected_language}').")
        
    if language_score < 70.0:
        recommendations.append("High character corruption/noise detected in OCR text. Apply image preprocessing/deskewing.")

    if not recommendations:
        recommendations.append("OCR extraction is high quality. All mandatory fields and text blocks extracted cleanly.")

    summary = (
        f"Document achieved a overall quality score of {overall_score}/100 ({grade}). "
        f"Completeness: {completeness_score}%, Field Accuracy: {field_accuracy_score}%, Language Quality: {language_score}%."
    )

    return QualityScoreBreakdown(
        overall_score=overall_score,
        completeness_score=completeness_score,
        field_accuracy_score=field_accuracy_score,
        language_score=language_score,
        grade=grade,
        summary=summary,
        recommendations=recommendations
    )
