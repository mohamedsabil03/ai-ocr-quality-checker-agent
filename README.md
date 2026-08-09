# AI OCR Quality Checker (SLM Agent)

A production-grade AI-powered OCR Quality Checker system built with **FastAPI**, Small Language Models (**Qwen3-4B-Instruct** / **Phi-4 Mini Instruct**), dedicated diagnostic tools, a calibrated 0–100 Scoring Engine, and an interactive Web Dashboard.

---

## Architecture Overview

```
                  ┌─────────────────┐
                  │    FastAPI      │
                  │   /ocr/check    │
                  └────────┬────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   OCR Quality Agent │
                │ Qwen / Phi-4 Mini   │
                └──────────┬──────────┘
                           │
                     Tool Calling
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     Missing Text     Mandatory Fields   Language
        Tool               Tool            Tool
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                 ┌──────────────────┐
                 │ Scoring Engine    │
                 │     0–100         │
                 └─────────┬────────┘
                           ▼
                 ┌──────────────────┐
                 │ Structured JSON  │
                 │ API Response     │
                 └─────────┬────────┘
```

---

## Directory Structure

```
ai-ocr-quality-checker/
│
├── app/
│   ├── main.py              # FastAPI endpoints & web router
│   ├── schemas.py           # Pydantic data schemas
│   ├── static/
│   │   └── index.html       # Web UI Dashboard
│   │
│   ├── agent/
│   │   ├── ocr_agent.py     # OCR Agent orchestrator
│   │   ├── prompts.py       # CoT & System prompts
│   │   └── model_loader.py  # Qwen3 & Phi-4 Mini SLM model loader
│   │
│   ├── tools/
│   │   ├── missing_text.py      # Missing text / omissions tool
│   │   ├── mandatory_fields.py  # Mandatory fields extractor tool
│   │   └── language_check.py    # Language & corruption check tool
│   │
│   ├── scoring/
│   │   └── quality_score.py # 0–100 Weighted scoring engine
│   │
│   └── utils/
│       └── helpers.py       # Fuzzy matching & text utilities
│
├── models/
│   ├── Qwen3-4B-Instruct/   # Local Qwen3 model weights
│   └── Phi-4 Mini Instruct/ # Local Phi-4 Mini model weights
│
├── dataset/
│   ├── test.json            # Test case suite
│   └── evaluation.json      # Benchmark evaluation criteria
│
├── tests/
│   ├── test_tools.py        # Unit tests for tools & scoring
│   └── test_api.py          # FastAPI integration tests
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Installation & Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run FastAPI Server**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Access Interactive Web UI Dashboard**:
Open your browser and navigate to: [http://localhost:8000](http://localhost:8000)

---

## API & Swagger Documentation

FastAPI provides automatic interactive Swagger UI and ReDoc documentation:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs) (Interactive API testing & schema explorer)
- **ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc) (Structured API documentation)

### Core Endpoints

#### 1. `POST /ocr/check`
Evaluates an OCR text payload against mandatory fields, language targets, and ground truth reference text.

**Request Payload**:
```json
{
  "ocr_text": "ACME LOGISTICS INC.\nINVOICE #INV-2026-8891\nDate: 2026-08-01\nTotal Amount: $4,950.00",
  "ground_truth_text": "ACME LOGISTICS INC.\nINVOICE #INV-2026-8891\nDate: 2026-08-01\nTotal Amount: $4,950.00",
  "mandatory_fields": ["Invoice Number", "Date", "Total Amount"],
  "target_language": "en",
  "model_name": "qwen3"
}
```

**Response**:
```json
{
  "id": "c1f7a635-430c-4fa6-879e-b5f7b4ee7ecb",
  "timestamp": "2026-08-08T21:35:00Z",
  "model_used": "qwen3",
  "score": {
    "overall_score": 100.0,
    "completeness_score": 100.0,
    "field_accuracy_score": 100.0,
    "language_score": 100.0,
    "grade": "EXCELLENT",
    "summary": "Document achieved a overall quality score of 100.0/100 (EXCELLENT).",
    "recommendations": ["OCR extraction is high quality."]
  },
  "tool_results": { ... },
  "agent_reasoning": "Agent Execution Log (Qwen3-4B-Instruct): ..."
}
```

### 2. `GET /health`
Returns service status and active SLM model load state.

### 3. `GET /models`
Lists available model weights (Qwen3 & Phi-4 Mini).

---

##  Running Automated Tests

Execute unit and integration test suites using `pytest`:

```bash
pytest tests/ -v
```
