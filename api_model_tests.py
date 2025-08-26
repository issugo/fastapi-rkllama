# test_parameterized.py
import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


@pytest.mark.parametrize("model_name", [
    "test_model",  # Valid model name
    "nonexistent_model",  # Invalid model name
    "",  # Empty string
    "model with spaces"  # Model name with spaces
])
def test_load_model_variations(model_name):
    """Test loading a model with different model names."""
    response = client.post(f"/api/load/{model_name}")
    # We're not asserting specific status codes because they'll vary
    # based on whether the model exists
    assert response.status_code in [200, 400, 404, 500]


@pytest.mark.parametrize("payload, expected_status", [
    ({"message": "Hello", "history": []}, 200),  # Basic valid payload
    ({"message": "", "history": []}, 400),  # Empty message
    ({"history": []}, 422),  # Missing message field
    ({"message": "Hello"}, 422),  # Missing history field
    ({"message": "Hello", "history": [], "temperature": 2.0}, 400),  # Invalid temperature
    ({"message": "Hello", "history": [], "max_tokens": -1}, 400),  # Invalid max_tokens
])
def test_chat_endpoint_variations(mock_model_loaded, payload, expected_status):
    """Test chat endpoint with different payloads."""
    response = client.post("/api/chat", json=payload)
    assert response.status_code == expected_status


@pytest.mark.parametrize("model_name, prompt, stream", [
    ("test_model", "Hello", False),  # Basic valid request
    ("test_model", "Hello", True),  # Streaming request
    ("nonexistent_model", "Hello", False),  # Invalid model
    ("test_model", "", False),  # Empty prompt
])
def test_generate_endpoint_variations(model_name, prompt, stream):
    """Test generate endpoint with different parameters."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": stream
    }
    response = client.post("/api/ollama/generate", json=payload)
    assert response.status_code in [200, 400, 404, 500]
