import re

OLLAMA_QUANT_FORMAT = ["Q4_0", "Q4_K_M", "Q8_0", "Q8_K_M"]

RK_QUANT_FORMAT = [
    "w4a16",
    "w4a16_g32",
    "w4a16_g64",
    "w4a16_g128",
    "w8a8",
    "w8a8_g128",
    "w8a8_g256",
    "w8a8_g512",
]


ollama_quant_mapping = {
    "Q4_0": "w4a16",
    "Q4_K_M": "w4a16_g128",
    "Q8_0": "w8a8",
    "Q8_K_M": "w8a8_g512",
}

# validation process
for ollama_quant_format, rk_quant_format in ollama_quant_mapping.items():
    assert (
        ollama_quant_format in OLLAMA_QUANT_FORMAT
    ), f"Missing {ollama_quant_format} in OLLAMA_QUANT_FORMAT"
    assert (
        rk_quant_format in RK_QUANT_FORMAT
    ), f"Missing {rk_quant_format} in RK_QUANT_FORMAT"

quant_mapping = {
    "w4a16": "Q4_0",
    "w4a16_g32": "Q4_K_M",
    "w4a16_g64": "Q4_K_M",
    "w4a16_g128": "Q4_K_M",
    "w8a8": "Q8_0",
    "w8a8_g128": "Q8_K_M",
    "w8a8_g256": "Q8_K_M",
    "w8a8_g512": "Q8_K_M",
}

# validation process
for rk_quant_format, ollama_quant_format in quant_mapping.items():
    assert (
        ollama_quant_format in OLLAMA_QUANT_FORMAT
    ), f"Missing {ollama_quant_format} in OLLAMA_QUANT_FORMAT"
    assert (
        rk_quant_format in RK_QUANT_FORMAT
    ), f"Missing {rk_quant_format} in RK_QUANT_FORMAT"


quant_patterns = [
    ("w4a16", r"w4a16(?!_g)"),
    ("w4a16_g32", r"w4a16_g32"),
    ("w4a16_g64", r"w4a16_g64"),
    ("w4a16_g128", r"w4a16_g128"),
    ("w8a8", r"w8a8(?!_g)"),
    ("w8a8_g128", r"w8a8_g128"),
    ("w8a8_g256", r"w8a8_g256"),
    ("w8a8_g512", r"w8a8_g512"),
]

# validation process
for rk_quant_format, rk_quant_format_pattern in quant_patterns:
    assert (
        rk_quant_format in RK_QUANT_FORMAT
    ), f"Missing {rk_quant_format} in RK_QUANT_FORMAT"
    assert re.match(
        rk_quant_format_pattern, rk_quant_format
    ), f"Mismatch {rk_quant_format} with pattern {rk_quant_format_pattern}"


def get_quant_pattern(quant_format):
    for quant_pattern, pattern in quant_patterns:
        if quant_pattern == quant_format:
            return pattern
    return None
