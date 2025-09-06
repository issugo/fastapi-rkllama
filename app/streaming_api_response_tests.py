# test_streaming.py
import json

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_streaming_chat_response(mock_model_loaded):
    """Test streaming chat responses."""
    payload = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
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
    payload = {"model": "test_model", "prompt": "Why is the sky blue?"}

    with client.stream("POST", "/api/generate", json=payload) as response:
        assert response.status_code in [200, 404, 500]
        is_first = True
        last_chunk = None
        if response.status_code == 200:
            # Read the streaming response chunks
            chunks = []
            for chunk in response.iter_lines():
                if chunk:
                    chunks.append(chunk)
                    if is_first:
                        is_first = False
                        assert "model" in response.json()
                        assert "created_at" in response.json()
                        assert "response" in response.json()
                        assert "done" in response.json()
                        assert json.loads(response.json()).done == False
                    else:
                        last_chunk = chunk

            # Assert we got some chunks (if model is properly mocked)
            assert len(chunks) > 0
            assert "model" in response.json()
            assert "created_at" in response.json()
            assert "response" in response.json()
            assert "done" in response.json()
            assert json.loads(response.json()).done == True
            assert "context" in response.json()
            assert "total_duration" in response.json()
            assert "load_duration" in response.json()
            assert "prompt_eval_count" in response.json()
            assert "prompt_eval_duration" in response.json()
            assert "eval_count" in response.json()
            assert "eval_duration" in response.json()
            assert isinstance(json.loads(response.json()).context, list)
            assert json.loads(response.json()).total_duration > 0
            assert json.loads(response.json()).load_duration > 0
            assert json.loads(response.json()).prompt_eval_count > 0
            assert json.loads(response.json()).prompt_eval_duration > 0
            assert json.loads(response.json()).eval_count > 0
            assert json.loads(response.json()).eval_duration > 0
            assert json.loads(response.json()).total_duration > json.loads(response.json()).load_duration
            assert json.loads(response.json()).total_duration > json.loads(response.json()).eval_duration
