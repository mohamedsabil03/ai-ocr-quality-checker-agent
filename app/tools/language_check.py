import re
from typing import Optional
from app.schemas import LanguageCheckResult

# Try importing langdetect safely
try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "en": "en",
    "spanish": "es",
    "es": "es",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "italian": "it",
    "it": "it",
    "portuguese": "pt",
    "pt": "pt",
    "chinese": "zh",
    "zh": "zh",
    "japanese": "ja",
    "ja": "ja"
}


def normalize_lang_code(code_or_name: str) -> str:
    """Normalizes language name or ISO code to standard 2-letter ISO code."""
    if not code_or_name:
        return "en"
    clean = code_or_name.strip().lower()
    return LANGUAGE_NAME_TO_CODE.get(clean, clean[:2])


def calculate_corruption_ratio(text: str) -> float:
    """Calculates ratio of corrupted/unusual OCR symbols and replacement chars."""
    if not text:
        return 1.0
    
    # Symbols common in corrupted OCR output
    corrupted_pattern = r'[\x00-\x08\x0b\x0c\x0e-\x1f│┤┐└┴┬├┼║═\^\~\`\{\}\\\|\<\>]'
    corrupted_matches = re.findall(corrupted_pattern, text)
    
    # Calculate non-alphanumeric/non-standard char ratio
    total_chars = len(text)
    corrupted_chars = len(corrupted_matches)
    
    return min(1.0, corrupted_chars / total_chars) if total_chars > 0 else 0.0


def run_language_check_tool(ocr_text: str, target_language: Optional[str] = "en") -> LanguageCheckResult:
    """
    Detects language of OCR text, matches against target language, and checks for character degradation.
    """
    expected_code = normalize_lang_code(target_language or "en")
    
    if not ocr_text or not ocr_text.strip():
        return LanguageCheckResult(
            detected_language="unknown",
            expected_language=expected_code,
            is_match=False,
            confidence=0.0,
            lang_score=0.0,
            details="OCR text is empty; language detection impossible."
        )

    clean_text = ocr_text.strip()
    detected_code = "en"
    confidence = 0.85
    
    # Run language detection
    if LANGDETECT_AVAILABLE and len(clean_text) > 10:
        try:
            predictions = detect_langs(clean_text)
            if predictions:
                top_pred = predictions[0]
                detected_code = top_pred.lang
                confidence = round(top_pred.prob, 4)
        except Exception:
            detected_code = "en"
            confidence = 0.70
    else:
        # Simple heuristic fallback
        if re.search(r'\b(the|and|invoice|total|date|amount|price)\b', clean_text, re.IGNORECASE):
            detected_code = "en"
        elif re.search(r'\b(el|la|factura|fecha|total|precio)\b', clean_text, re.IGNORECASE):
            detected_code = "es"
        elif re.search(r'\b(le|la|facture|date|total|prix)\b', clean_text, re.IGNORECASE):
            detected_code = "fr"
        elif re.search(r'\b(der|die|das|rechnung|datum|betrag)\b', clean_text, re.IGNORECASE):
            detected_code = "de"

    is_match = (detected_code == expected_code)
    
    # Calculate corruption / gibberish metric
    corruption_ratio = calculate_corruption_ratio(clean_text)
    clarity_score = max(0.0, 1.0 - (corruption_ratio * 3.0))
    
    # Final language score formula
    match_score = 1.0 if is_match else 0.5
    lang_score = round(0.6 * match_score + 0.4 * clarity_score, 4)

    details = (
        f"Language detection result: Detected '{detected_code}' (Confidence: {confidence * 100:.1f}%). "
        f"Target language: '{expected_code}'. Match status: {is_match}. "
        f"Character integrity clarity score: {clarity_score * 100:.1f}%."
    )

    return LanguageCheckResult(
        detected_language=detected_code,
        expected_language=expected_code,
        is_match=is_match,
        confidence=confidence,
        lang_score=lang_score,
        details=details
    )
