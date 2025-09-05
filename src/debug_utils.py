import json

from core import config


def check_response_format(response_text):
    """
    Check if a response stream has the correct format

    Args:
        response_text: String containing newline-separated JSON responses

    Returns:
        List of issues found, or empty list if format is correct
    """
    issues = []

    if not response_text:
        return ["Empty response stream"]

    lines = response_text.strip().split("\n")

    # Check if we have at least one line
    if not lines:
        return ["No lines found in response"]

    # Parse each line as JSON
    parsed_chunks = []
    for i, line in enumerate(lines):
        try:
            chunk = json.loads(line)
            parsed_chunks.append(chunk)
        except json.JSONDecodeError:
            issues.append(f"Line {i + 1} is not valid JSON: {line[:50]}...")

    if not parsed_chunks:
        return ["No valid JSON chunks found in response"]

    # Check if the last chunk has done=True
    last_chunk = parsed_chunks[-1]
    if not config.get("done", False):
        issues.append("Last chunk does not have 'done' set to true")

    # Check for consistency in response format
    first_chunk = parsed_chunks[0]

    # Check if using generate or chat format
    is_generate_format = "response" in first_chunk
    is_chat_format = "message" in first_chunk and isinstance(
        config.get("message", {}), dict
    )

    if not (is_generate_format or is_chat_format):
        issues.append(
            f"First chunk has neither 'response' nor 'message.content' field: {first_chunk}"
        )

    # Check consistent format through all chunks
    for i, chunk in enumerate(parsed_chunks):
        if is_generate_format and "response" not in chunk:
            issues.append(f"Chunk {i + 1} missing 'response' field in generate format")

        if is_chat_format:
            if "message" not in chunk:
                issues.append(f"Chunk {i + 1} missing 'message' field in chat format")
            elif not isinstance(chunk["message"], dict):
                issues.append(f"Chunk {i + 1} has 'message' that is not a dictionary")
            elif "content" not in chunk["message"]:
                issues.append(f"Chunk {i + 1} missing 'message.content' field")

    return issues
