ollama_quant_mapping = {
    'Q4_0': 'w4a16',
    'Q4_K_M': 'w4a16_g128',
    'Q8_0': 'w8a8',
    'Q8_K_M': 'w8a8_g512'
}

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

def get_quant_pattern(quant_format):
    for quant_pattern, pattern in quant_patterns:
        if quant_pattern == quant_format:
            return pattern
    return None

