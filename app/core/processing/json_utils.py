import json
import re


def extract_json(text):
    """Extract JSON from text that might contain non-JSON content"""

    # First look for JSON in code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    code_matches = re.findall(code_block_pattern, text)

    for potential_json in code_matches:
        try:
            parsed = json.loads(potential_json)
            return potential_json.strip(), parsed
        except json.JSONDecodeError:
            continue

    # If no valid JSON in code blocks, try to find JSON-like content directly
    json_pattern = r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})"
    json_matches = re.findall(json_pattern, text)

    for potential_json in json_matches:
        try:
            parsed = json.loads(potential_json)
            return potential_json.strip(), parsed
        except json.JSONDecodeError:
            continue

    # Try with more lenient pattern
    more_lenient_pattern = r"\{[\s\S]*?\}"
    lenient_matches = re.findall(more_lenient_pattern, text)

    for potential_json in lenient_matches:
        # Clean up the text
        cleaned = re.sub(r'[^\{\}\[\],:."\'0-9a-zA-Z_\s-]', "", potential_json)
        cleaned = cleaned.replace("'", '"')  # Replace single quotes with double quotes

        try:
            parsed = json.loads(cleaned)
            return cleaned.strip(), parsed
        except json.JSONDecodeError:
            continue

    # No valid JSON found
    return None, None


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))
