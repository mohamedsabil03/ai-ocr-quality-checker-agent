import re
from typing import Optional
from app.schemas import MissingTextResult
from app.utils.helpers import compute_similarity, find_missing_text_segments, normalize_text


def run_missing_text_tool(ocr_text: str, ground_truth_text: Optional[str] = None) -> MissingTextResult:
    """
    Evaluates missing text, omissions, and completeness in OCR text output.
    Can operate against a Ground Truth reference text or via structural heuristics.
    """
    if not ocr_text or not ocr_text.strip():
        if ground_truth_text and ground_truth_text.strip():
            missing_snippets = [line.strip() for line in ground_truth_text.split('\n') if line.strip()]
            return MissingTextResult(
                similarity_score=0.0,
                missing_snippets=missing_snippets,
                missing_count=len(missing_snippets),
                details="OCR text is completely blank while ground truth reference exists."
            )
        return MissingTextResult(
            similarity_score=0.0,
            missing_snippets=["[EMPTY OCR TEXT]"],
            missing_count=1,
            details="OCR text payload is empty."
        )

    # 1. Ground Truth comparison mode
    if ground_truth_text and ground_truth_text.strip():
        similarity = compute_similarity(ground_truth_text, ocr_text)
        missing_snippets = find_missing_text_segments(ground_truth_text, ocr_text)
        
        details = (
            f"Ground Truth comparison complete. Similarity ratio: {similarity * 100:.1f}%. "
            f"Found {len(missing_snippets)} missing segment(s)."
        )
        if missing_snippets:
            details += f" Key omitted items: '{missing_snippets[0][:50]}...'"
            
        return MissingTextResult(
            similarity_score=similarity,
            missing_snippets=missing_snippets,
            missing_count=len(missing_snippets),
            details=details
        )

    # 2. Reference-free heuristic completeness mode
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
    suspicious_truncations = []
    
    for idx, line in enumerate(lines):
        # Check if line ends mid-sentence or mid-word without punctuation
        if re.search(r'[a-zA-Z0-9],?\s*$', line) and idx < len(lines) - 1:
            next_line = lines[idx + 1]
            if next_line and next_line[0].islower():
                pass # multi-line sentence continuation
        
        # Incomplete key-value pattern (e.g. "Total Amount:")
        if re.search(r':\s*$', line):
            suspicious_truncations.append(f"Incomplete value for field line: '{line}'")
            
        # Corrupted line artifacts
        if re.search(r'[^\w\s.,;:$\-/#()]{4,}', line):
            suspicious_truncations.append(f"Corrupted noise pattern in line: '{line[:40]}'")

    # Calculate heuristic score based on structure
    base_score = 1.0
    if suspicious_truncations:
        base_score = max(0.4, 1.0 - (len(suspicious_truncations) * 0.15))

    details = (
        f"Reference-free completeness analysis. Structural integrity score: {base_score * 100:.1f}%. "
        f"Detected {len(suspicious_truncations)} potential structural anomaly/truncation site(s)."
    )

    return MissingTextResult(
        similarity_score=round(base_score, 4),
        missing_snippets=suspicious_truncations,
        missing_count=len(suspicious_truncations),
        details=details
    )
