import re

MODEL_SPECS = {
    "qwen2": (4096, [r"(?i)qwen2"]),
    "qwen3": (4096, [r"(?i)qwen3"]),
    "mistral": (8192, [r"(?i)mistral"]),
    "llama3": (4096, [r"(?i)llama[-_]?3"]),
    "llama2": (4096, [r"(?i)llama[-_]?2"]),
    "gemma": (4096, [r"(?i)gemma"]),
    "deepseek": (4096, [r"(?i)deepseek"]),
    "phi": (2048, [r"(?i)phi"]),
    "baichuan": (2048, [r"(?i)baichuan"]),
    "yi": (2048, [r"(?i)yi"]),
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


# "qwen2.context_length": 32768,
# ["llama", "llama2", "llama3"] f"{family}.context_length": 4096,
# "mistral.context_length": 8192,
def default_context_length(family: str):
    for name, (ctx_len, _) in MODEL_SPECS.items():
        if name == family:
            return ctx_len
    return 4096


MODEL_WITH_TOOLS = ["qwen2", "qwen3", "phi", "llama3", "mistral"]

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

MODEL_ARCHITECTURE_MAPPING = {
    "llama": ["llama", "llama2", "llama3"],
    "mistral": ["mistral"],
    "qwen": ["qwen2", "qwen3"],
    "deepseek": ["deepseek"],
    "phi": ["phi"],
    "gemma": ["gemma"],
    "baichuan": ["baichuan"],
    "yi": ["yi"],
}

# validation process
for model_family, model_list in MODEL_ARCHITECTURE_MAPPING.items():
    assert (
        model_family in MODEL_ARCHITECTURES
    ), f"Missing {model_family} in MODEL_ARCHITECTURES"
    for model in model_list:
        assert model in MODEL_SPECS, f"Missing {model} in MODEL_SPECS"
# validation process
for model_with_tools in MODEL_WITH_TOOLS:
    found = False
    for model_family, model_list in MODEL_ARCHITECTURE_MAPPING.items():
        if model_with_tools == model_family:
            found = True
        if model_with_tools in model_list:
            found = True
    assert found, f"Missing {model_with_tools} in MODEL_ARCHITECTURE_MAPPING"


# HF license ID to readable name
LICENSE_NAME_MAPPING = {
    "apache-2.0": "Apache 2.0",
    "mit": "MIT",
    "cc-by-4.0": "Creative Commons Attribution 4.0",
    "cc-by-sa-4.0": "Creative Commons Attribution-ShareAlike 4.0",
    "cc-by-nc-4.0": "Creative Commons Attribution-NonCommercial 4.0",
    "cc-by-nc-sa-4.0": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
}

RK_TAGS_LIST = ["rk3588", "rk3576", "rkllm", "rknn", "rockchip"]

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

MODELFILE_NAME: str = "Modelfile"

B_PARAM_SIZE_PATTERN = r"(\d+\.?\d*)([bB])"
M_PARAM_SIZE_PATTERN = r"(\d+\.?\d*)([mM])"

UNKNOWN_VAL_STR = "Unknown"
MODELS_STORAGE_DIR = "models"

APACHE2_COMMON_LICENSE = "apache-2.0"
MIT_COMMON_LICENSE = "mit"
QWEN_COMMON_LICENSE = "qwen-research"

COMMON_LICENSES = [APACHE2_COMMON_LICENSE, MIT_COMMON_LICENSE, QWEN_COMMON_LICENSE]

COMMON_LICENSE = {
    APACHE2_COMMON_LICENSE: "apache",
    MIT_COMMON_LICENSE: "mit",
    QWEN_COMMON_LICENSE: "qwen-research",
}

# validation process
for common_lic in COMMON_LICENSES:
    assert common_lic in COMMON_LICENSE, f"Missing {common_lic} in COMMON_LICENSE"
for common_lic in COMMON_LICENSE.keys():
    assert common_lic in COMMON_LICENSES, f"Missing {common_lic} in COMMON_LICENSES"


def validate_model_id(model_id: str):
    """Check that FROM contains a valid model name or path to a model file."""
    if len(model_id.split("/")) == 0 and len(model_id.split(":")) == 0:
        raise ValueError(
            "FROM and model_id must be a valid model name or path to a model file."
        )
    return model_id


def validate_from(from_str: str):
    return validate_model_id(from_str)


DEFAULT_SYSTEM = "Tu es un assistant artificiel."
DEFAULT_TEMPLATE = ""
