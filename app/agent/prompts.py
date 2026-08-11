"""
Prompts and templates for OCR Quality Agent (Qwen3-4B-Instruct) Tool-Calling Loop.
"""

TOOL_DEFINITIONS = [
    {
        "name": "missing_text_tool",
        "description": "Evaluates line omissions, missing text blocks, and string similarity against reference ground truth.",
        "parameters": {
            "type": "object",
            "properties": {
                "ocr_text": {"type": "string", "description": "Raw OCR text to evaluate"},
                "ground_truth_text": {"type": "string", "description": "Optional ground truth reference text"}
            },
            "required": ["ocr_text"]
        }
    },
    {
        "name": "mandatory_fields_tool",
        "description": "Checks presence of mandatory fields (e.g. Invoice Number, Date, Total Amount, Vendor) in the OCR text.",
        "parameters": {
            "type": "object",
            "properties": {
                "ocr_text": {"type": "string", "description": "Raw OCR text to analyze"},
                "mandatory_fields": {"type": "array", "items": {"type": "string"}, "description": "List of field names expected"}
            },
            "required": ["ocr_text"]
        }
    },
    {
        "name": "language_check_tool",
        "description": "Detects primary language of OCR text, checks against expected language target, and measures character corruption.",
        "parameters": {
            "type": "object",
            "properties": {
                "ocr_text": {"type": "string", "description": "Raw OCR text"},
                "target_language": {"type": "string", "description": "Expected language ISO code (e.g. 'en', 'es', 'de')"}
            },
            "required": ["ocr_text"]
        }
    }
]


REACT_SYSTEM_PROMPT = """You are an autonomous AI OCR Quality Agent powered by Small Language Models (Qwen3-4B-Instruct).
Your job is to audit OCR text extractions using available diagnostic tools before rendering a final quality assessment.

AVAILABLE DIAGNOSTIC TOOLS:
1. `missing_text_tool`: Analyzes missing text segments and sequence similarity.
2. `mandatory_fields_tool`: Checks presence of required key fields (Invoice Number, Date, Total Amount, etc.).
3. `language_check_tool`: Detects document language and character corruption noise.

TOOL INVOCATION FORMAT:
To call a tool, respond with a JSON object in this format:
```json
{
  "action": "tool_call",
  "tool_name": "mandatory_fields_tool",
  "arguments": {
    "ocr_text": "...",
    "mandatory_fields": ["Invoice Number", "Date", "Total Amount"]
  }
}
```

FINAL ANSWER FORMAT:
When you have gathered all necessary tool observations, respond with your final assessment in this format:
```json
{
  "action": "final_answer",
  "reasoning": "Step-by-step diagnostic reasoning...",
  "verdict": "GOOD"
}
```
"""


def format_react_prompt(ocr_text: str, conversation_history: list, model_name: str) -> str:
    """Formats prompt payload with conversation history for ReAct agent loop."""
    prompt = f"<|im_start|>system\n{REACT_SYSTEM_PROMPT}\n<|im_end|>\n"
    prompt += f"<|im_start|>user\nPlease evaluate this OCR Text payload:\n\"\"\"\n{ocr_text}\n\"\"\"\n<|im_end|>\n"
    for turn in conversation_history:
        role = turn.get("role", "assistant")
        content = turn.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt
