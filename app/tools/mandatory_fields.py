from typing import List, Optional
from app.schemas import MandatoryFieldsResult
from app.utils.helpers import check_field_presence


DEFAULT_MANDATORY_FIELDS = ["Invoice Number", "Date", "Total Amount"]


def run_mandatory_fields_tool(
    ocr_text: str, mandatory_fields: Optional[List[str]] = None
) -> MandatoryFieldsResult:
    """
    Evaluates whether required mandatory key fields are present in the OCR text output.
    """
    fields_to_check = mandatory_fields if mandatory_fields else DEFAULT_MANDATORY_FIELDS
    
    if not fields_to_check:
        return MandatoryFieldsResult(
            found_fields=[],
            missing_fields=[],
            field_presence_ratio=1.0,
            field_details={}
        )

    if not ocr_text or not ocr_text.strip():
        field_details = {field: False for field in fields_to_check}
        return MandatoryFieldsResult(
            found_fields=[],
            missing_fields=list(fields_to_check),
            field_presence_ratio=0.0,
            field_details=field_details
        )

    found = []
    missing = []
    field_details = {}

    for field in fields_to_check:
        present = check_field_presence(field, ocr_text)
        field_details[field] = present
        if present:
            found.append(field)
        else:
            missing.append(field)

    presence_ratio = len(found) / len(fields_to_check) if fields_to_check else 1.0

    return MandatoryFieldsResult(
        found_fields=found,
        missing_fields=missing,
        field_presence_ratio=round(presence_ratio, 4),
        field_details=field_details
    )
