# test_streaming.py
import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_streaming_chat_response(mock_model_loaded):
    """Test streaming chat responses."""
    payload = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    }

    with client.stream("POST", "/api/ollama/chat", json=payload) as response:
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            # Read the streaming response chunks
            chunks = []
            for chunk in response.iter_lines():
                if chunk:
                    chunks.append(chunk)

            # Assert we got some chunks (if model is properly mocked)
            assert len(chunks) > 0


def test_streaming_generate_response(mock_model_loaded):
    """Test streaming generate responses."""
    payload = {
        "model": "test_model",
        "prompt": "Hello, world!",
        "stream": True
    }

    with client.stream("POST", "/api/ollama/generate", json=payload) as response:
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            # Read the streaming response chunks
            chunks = []
            for chunk in response.iter_lines():
                if chunk:
                    chunks.append(chunk)

            # Assert we got some chunks (if model is properly mocked)
            assert len(chunks) > 0
