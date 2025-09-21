import re

MODEL_SPECS = {
    "qwen2": (4096, [r"(?i)qwen"]),
    "mistral": (4096, [r"(?i)mistral"]),
    "llama3": (4096, [r"(?i)llama[-_]?3"]),
    "llama2": (4096, [r"(?i)llama[-_]?2"]),
    "gemma": (4096, [r"(?i)gemma"]),
    "phi": (2048, [r"(?i)phi"]),
    "llama": (4096, []),  # fallback
}


def detect_family(text: str) -> str:
    return next(
        (
            name
            for name, (_, patterns) in MODEL_SPECS.items()
            for p in patterns
            if re.search(p, text)
        ),
        "llama",
    )

# HF license ID to readable name
LICENSE_NAME_MAPPING = {
    "apache-2.0": "Apache 2.0",
    "mit": "MIT",
    "cc-by-4.0": "Creative Commons Attribution 4.0",
    "cc-by-sa-4.0": "Creative Commons Attribution-ShareAlike 4.0",
    "cc-by-nc-4.0": "Creative Commons Attribution-NonCommercial 4.0",
    "cc-by-nc-sa-4.0": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
}

RK_TAGS_LIST = ["rk3588", "rk3576", "rkllm", "rockchip"]

LANGUAGE_DEFAULT = ["en"]

LANGUAGE_MULTILINGUAL_LIST = ["en", "zh", "fr", "de", "es", "ja"]

LANGUAGE_PATTERNS = {
    "english": "en",
    "chinese": "zh",
    "multilingual": None,  # Special case
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "japanese": "ja",
}

MODEL_ARCHITECTURES = {
    "llama": "llama",
    "mistral": "mistral",
    "qwen": "qwen",
    "deepseek": "deepseek",
    "phi": "phi",
    "gemma": "gemma",
    "baichuan": "baichuan",
    "yi": "yi",
}

MODELFILE_NAME:str = "Modelfile"
