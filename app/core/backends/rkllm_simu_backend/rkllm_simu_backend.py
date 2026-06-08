import os
import json
import time
import logging
import requests
from transformers import AutoTokenizer

from core.backends.backend import Backend, BackendType
from core.model.Model import Model
from core.model.ModelConfig import FullModelParameters
from core.processing.BaseDomainId import BaseDomainId
from core.processing.variables import global_text

logger = logging.getLogger("rkllama.rkllm_simu")


class RkllmSimuBackend(Backend):
    def __init__(
        self,
        model: Model,
        options: FullModelParameters,
        base_domain_id: BaseDomainId,
        prompt_cache_path=None,
        lora_model_path=None,
    ):
        super().__init__(BackendType.RKLLM)
        self.model = model
        self.options = options
        self.base_domain_id = base_domain_id

        logger.info(
            f"Initialized RkllmSimuBackend for model {model.id} using Gemini simulation."
        )

        # Determine the Hugging Face model repository to load the tokenizer
        self.hf_path = None
        if hasattr(model, "model_path") and getattr(
            model.model_path, "huggingface_path", None
        ):
            self.hf_path = model.model_path.huggingface_path
        elif hasattr(model, "model_info") and getattr(model.model_info, "id", None):
            self.hf_path = model.model_info.id
        else:
            self.hf_path = model.id

        logger.debug(f"Determined tokenizer path: {self.hf_path}")
        self.tokenizer = None
        try:
            if self.hf_path:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_path, trust_remote_code=True
                )
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer from {self.hf_path}: {e}. Falling back to default decoding."
            )

    def run(self, prompt_tokens):
        logger.debug(f"RkllmSimuBackend run called with {len(prompt_tokens)} tokens.")

        # 1. Decode prompt tokens to text
        prompt_text = ""
        if self.tokenizer:
            try:
                prompt_text = self.tokenizer.decode(
                    prompt_tokens, skip_special_tokens=True
                )
            except Exception as e:
                logger.error(f"Failed to decode prompt tokens: {e}")
                prompt_text = "Hello"
        else:
            prompt_text = f"Simulated prompt with {len(prompt_tokens)} tokens."

        # 2. Get API key and call Gemini LLM API
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning(
                "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is configured. Simulating response."
            )
            # Emulate token-by-token generation for testing
            simulated_text = (
                f'[Simulated Gemini Response] Your prompt was: "{prompt_text}". '
                "Please configure GEMINI_API_KEY or GOOGLE_API_KEY to retrieve live responses from Gemini."
            )
            for word in simulated_text.split(" "):
                global_text.append(word + " ")
                time.sleep(0.02)
            return

        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?key={api_key}"

        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}

        try:
            response = requests.post(url, json=payload, stream=True, timeout=30)
            response.raise_for_status()

            for chunk in response.iter_lines():
                if not chunk:
                    continue
                chunk_str = chunk.decode("utf-8").strip()
                if chunk_str.startswith("data:"):
                    chunk_str = chunk_str[5:].strip()
                if not chunk_str:
                    continue
                if chunk_str.startswith("["):
                    chunk_str = chunk_str[1:].strip()
                if chunk_str.endswith("]"):
                    chunk_str = chunk_str[:-1].strip()
                if chunk_str.endswith(","):
                    chunk_str = chunk_str[:-1].strip()

                try:
                    data = json.loads(chunk_str)
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    global_text.append(text)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            global_text.append(f"Error calling Gemini API: {str(e)}")

    def abort(self):
        logger.info("RkllmSimuBackend: abort called")

    def clear_cache(self):
        logger.info("RkllmSimuBackend: clear_cache called")

    def release(self):
        logger.info("RkllmSimuBackend: release called")
