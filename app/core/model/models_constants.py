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
