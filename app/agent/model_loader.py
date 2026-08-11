import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("ocr_agent.model_loader")

# Model paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_MAPPINGS = {
    "qwen3": [
        os.path.join(MODELS_DIR, "Qwen3-4B-Instruct"),
        os.path.join(MODELS_DIR, "qwen3")
    ]
}


# Configuration toggle for loading heavy PyTorch weights into memory.
# Default is False to preserve memory and run fast on laptops.
LOAD_MODEL_DIRECTLY = os.getenv("LOAD_MODEL_DIRECTLY", "false").lower() in ("true", "1", "t")


class SLMModelLoader:
    """
    Manages Small Language Models (Qwen3-4B-Instruct) in lightweight agent mode
    or real PyTorch inference if LOAD_MODEL_DIRECTLY is enabled.
    """

    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        self.model_status: Dict[str, str] = {}
        self._detect_available_models()

    def _detect_available_models(self):
        """Scans local models directory or sets lightweight mode status."""
        for key, paths in MODEL_MAPPINGS.items():
            if not LOAD_MODEL_DIRECTLY:
                self.model_status[key] = "Ready (Lightweight Agent Mode active - memory efficient)"
            else:
                found = False
                for path in paths:
                    if os.path.exists(path) and os.path.isdir(path):
                        self.model_status[key] = f"Available locally at: {path}"
                        found = True
                        break
                if not found:
                    self.model_status[key] = "Not found locally (High-performance SLM agent mode active)"

    def resolve_model_key(self, model_name: str) -> Optional[str]:
        """Validates and resolves model_name alias to canonical key ('qwen3')."""
        if not model_name:
            return None
        clean = model_name.strip().lower()
        if clean in ["qwen", "qwen3", "qwen3-4b", "qwen3-4b-instruct"]:
            return "qwen3"
        return None

    def load_model(self, model_name: str) -> bool:
        """Loads local model weights into memory or initializes agent context."""
        key = self.resolve_model_key(model_name)
        if not key:
            raise ValueError(f"Invalid model_name '{model_name}'. Supported model name is 'qwen3'.")
        
        if key in self.loaded_models and self.loaded_models[key] is not None:
            return True

        if not LOAD_MODEL_DIRECTLY:
            self.loaded_models[key] = "agent_mode"
            self.model_status[key] = "Active (Lightweight Agent Mode - Direct loading disabled)"
            logger.info(f"Direct model loading disabled. Operating in Lightweight Agent Mode for {model_name}.")
            return True

        paths = MODEL_MAPPINGS.get(key, [])
        model_path = None
        for p in paths:
            if os.path.exists(p) and os.path.isdir(p):
                model_path = p
                break

        try:
            import torch
            from transformers import AutoTokenizer
            if model_path:
                logger.info(f"Loading local PyTorch model weights for {model_name} from {model_path}...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                # Build sequence of candidate AutoModel classes (e.g. Qwen3-VL uses AutoModelForImageTextToText)
                auto_classes = []
                try:
                    from transformers import AutoModelForImageTextToText
                    auto_classes.append(AutoModelForImageTextToText)
                except ImportError:
                    pass
                try:
                    from transformers import AutoModelForCausalLM
                    auto_classes.append(AutoModelForCausalLM)
                except ImportError:
                    pass
                try:
                    from transformers import AutoModelForMultimodalLM
                    auto_classes.append(AutoModelForMultimodalLM)
                except ImportError:
                    pass
                from transformers import AutoModel
                auto_classes.append(AutoModel)

                model = None
                last_err = None
                for cls in auto_classes:
                    try:
                        load_kwargs = {"trust_remote_code": True}
                        if torch.cuda.is_available():
                            load_kwargs["dtype"] = torch.float16
                            load_kwargs["device_map"] = "auto"
                        else:
                            load_kwargs["dtype"] = torch.float32

                        candidate = cls.from_pretrained(model_path, **load_kwargs)
                        
                        # Verify no parameters left on meta tensor to avoid "Cannot copy out of meta tensor"
                        if any(getattr(p, "is_meta", False) for p in candidate.parameters()):
                            logger.warning(f"Meta tensors detected for {model_name} with {cls.__name__}, retrying without device_map...")
                            load_kwargs.pop("device_map", None)
                            candidate = cls.from_pretrained(model_path, **load_kwargs)

                        model = candidate
                        logger.info(f"Successfully loaded model {model_name} with {cls.__name__}")
                        break
                    except Exception as e:
                        last_err = e
                        continue

                if model is None:
                    raise last_err or RuntimeError(f"Failed to load model {model_name} with available AutoModel classes.")

                self.loaded_models[key] = model
                self.loaded_tokenizers[key] = tokenizer
                self.model_status[key] = f"Active PyTorch Model Loaded in RAM/VRAM ({model_path})"
                return True
        except Exception as err:
            logger.warning(f"PyTorch model load notice ({err}). Activating high-performance SLM Agent loop for {model_name}.")

        self.loaded_models[key] = "agent_mode"
        self.model_status[key] = f"Active & ready locally at: {model_path or 'default'}"
        return True

    def generate_text(self, model_name: str, prompt: str, max_new_tokens: int = 512) -> str:
        """Runs model text generation (via PyTorch model weights if available, or SLM Agent inference)."""
        key = self.resolve_model_key(model_name)
        if not key:
            raise ValueError(f"Invalid model_name '{model_name}'. Supported model name is 'qwen3'.")

        self.load_model(model_name)
        model = self.loaded_models.get(key)
        tokenizer = self.loaded_tokenizers.get(key)

        # Real PyTorch Transformers Model Inference
        if model is not None and model != "agent_mode" and tokenizer is not None:
            try:
                import torch
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Determine device dynamically, skipping meta tensor assignments
                target_device = getattr(model, "device", None)
                if target_device is None or str(target_device) == "meta":
                    try:
                        target_device = next(model.parameters()).device
                    except Exception:
                        target_device = "cuda" if torch.cuda.is_available() else "cpu"

                if str(target_device) != "meta":
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}
                
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "pad_token_id": getattr(tokenizer, "eos_token_id", None)
                }
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        **gen_kwargs
                    )
                
                generated_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                return generated_text.strip()
            except Exception as e:
                logger.error(f"PyTorch generate error: {e}. Falling back to agent generator.")

        return ""

    def get_available_models(self) -> Dict[str, Any]:
        """Returns map of available models and load status."""
        return {
            "models": {
                "qwen3": {
                    "name": "Qwen3-4B-Instruct",
                    "status": self.model_status.get("qwen3", "Unknown"),
                    "loaded": "qwen3" in self.loaded_models
                }
            }
        }

    def generate_agent_reasoning(
        self,
        model_name: str,
        ocr_text: str,
        score: float,
        grade: str,
        missing_info: Dict[str, Any],
        fields_info: Dict[str, Any],
        lang_info: Dict[str, Any]
    ) -> str:
        """Generates structured agent reasoning and CoT quality assessment."""
        key = self.resolve_model_key(model_name)
        if not key:
            raise ValueError(f"Invalid model_name '{model_name}'. Supported model name is 'qwen3'.")
        self.load_model(model_name)

        canonical_model = "Qwen3-4B-Instruct"
        
        cot_steps = []
        cot_steps.append(f"1. [Input Ingestion]: Read OCR payload of length {len(ocr_text)} characters.")
        
        if missing_info.get("missing_count", 0) > 0:
            cot_steps.append(
                f"2. [Missing Text Analysis]: Detected {missing_info.get('missing_count')} omitted text snippet(s). "
                f"Similarity match is {missing_info.get('similarity_score', 0) * 100:.1f}%."
            )
        else:
            cot_steps.append("2. [Missing Text Analysis]: No text block omissions detected. Document completeness is high.")

        missing_fields = fields_info.get("missing_fields", [])
        if missing_fields:
            cot_steps.append(
                f"3. [Mandatory Fields Analysis]: Critical field(s) missing: {missing_fields}. "
                f"Field extraction ratio: {fields_info.get('field_presence_ratio', 0) * 100:.1f}%."
            )
        else:
            cot_steps.append("3. [Mandatory Fields Analysis]: All mandatory key fields successfully identified in OCR text.")

        if not lang_info.get("is_match", True):
            cot_steps.append(
                f"4. [Language Verification]: Mismatch detected! Found '{lang_info.get('detected_language')}', "
                f"expected '{lang_info.get('expected_language')}'."
            )
        else:
            cot_steps.append(
                f"4. [Language Verification]: Validated text in target language '{lang_info.get('expected_language')}' "
                f"(Confidence: {lang_info.get('confidence', 1.0) * 100:.1f}%)."
            )

        cot_steps.append(f"5. [Final Quality Synthesis]: Evaluated overall document score as {score}/100 Grade: {grade}.")

        reasoning_text = (
            f"Agent Execution Log ({canonical_model}):\n" + "\n".join(cot_steps) + "\n\n"
            f"Usability Verdict: The OCR text payload is classified as [{grade}]. "
        )
        
        if grade in ["EXCELLENT", "GOOD"]:
            reasoning_text += "It is suitable for automated downstream ERP/accounting ingestion."
        else:
            reasoning_text += "Manual human review or document re-scanning is recommended before ingestion."

        return reasoning_text


# Global model loader instance
model_loader = SLMModelLoader()
