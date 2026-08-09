import re
import difflib
from typing import List, Tuple, Dict, Any


def normalize_text(text: str) -> str:
    """Normalizes text by trimming whitespace, standardizing line breaks, and lowercasing."""
    if not text:
        return ""
    # Standardize line breaks & spaces
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def compute_similarity(s1: str, s2: str) -> float:
    """Computes sequence similarity ratio between two strings using difflib."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    norm1 = normalize_text(s1).lower()
    norm2 = normalize_text(s2).lower()
    
    matcher = difflib.SequenceMatcher(None, norm1, norm2)
    return round(matcher.ratio(), 4)


def find_missing_text_segments(ground_truth: str, ocr_text: str, min_len: int = 3) -> List[str]:
    """
    Identifies text segments present in ground truth but omitted in OCR output.
    """
    if not ground_truth:
        return []
    
    if not ocr_text:
        # Whole ground truth is missing
        lines = [line.strip() for line in ground_truth.split('\n') if len(line.strip()) >= min_len]
        return lines if lines else [ground_truth]
    
    gt_lines = [line.strip() for line in ground_truth.split('\n') if line.strip()]
    ocr_norm = normalize_text(ocr_text).lower()
    
    missing = []
    for line in gt_lines:
        line_norm = line.lower()
        if len(line_norm) < min_len:
            continue
        
        # Check direct substring
        if line_norm in ocr_norm:
            continue
        
        # Check fuzzy match against sliding windows or lines of OCR text
        best_match = 0.0
        for ocr_line in ocr_text.split('\n'):
            ocr_line_norm = ocr_line.strip().lower()
            if not ocr_line_norm:
                continue
            ratio = difflib.SequenceMatcher(None, line_norm, ocr_line_norm).ratio()
            if ratio > best_match:
                best_match = ratio
        
        if best_match < 0.65:
            missing.append(line)
            
    return missing


def build_field_patterns(field_name: str) -> List[str]:
    """Generates regex patterns for matching field names and key-value formats."""
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', field_name).lower()
    words = clean.split()
    
    patterns = [
        re.escape(field_name),
        r'\b' + r'[\s_\-]*'.join([re.escape(w) for w in words]) + r'\b',
    ]
    
    # Common field aliases
    aliases = {
        "invoice number": [r"inv\s*#?", r"invoice\s*no\.?", r"invoice\s*num", r"bill\s*no"],
        "date": [r"dt\.?", r"dated", r"invoice\s*date", r"issue\s*date"],
        "total amount": [r"total", r"grand\s*total", r"amount\s*due", r"net\s*payable", r"balance\s*due"],
        "tax": [r"vat", r"gst", r"sales\s*tax", r"tax\s*amount"],
        "phone": [r"tel", r"mobile", r"contact", r"phone\s*no"],
        "email": [r"e-mail", r"mail", r"contact\s*email"],
        "address": [r"location", r"street", r"billing\s*address", r"shipping\s*address"],
        "vendor": [r"company", r"merchant", r"supplier", r"biller", r"seller"],
        "customer": [r"client", r"bill\s*to", r"buyer", r"customer\s*name"]
    }
    
    clean_lower = clean.lower()
    for alias_key, alias_patterns in aliases.items():
        if alias_key in clean_lower or clean_lower in alias_key:
            patterns.extend(alias_patterns)
            
    return patterns


def check_field_presence(field_name: str, text: str) -> bool:
    """Checks whether a mandatory field (or its value representation) exists in the OCR text."""
    if not text:
        return False
    
    text_lower = text.lower()
    patterns = build_field_patterns(field_name)
    
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
            
    # Substring search as fallback
    if field_name.lower() in text_lower:
        return True
        
    return False


def extract_json_from_text(raw_text: str) -> str:
    """Extracts JSON string from markdown code block or raw string."""
    if not raw_text:
        return "{}"
    
    # Match ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Find first { and last }
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end+1].strip()
        
    return raw_text.strip()
