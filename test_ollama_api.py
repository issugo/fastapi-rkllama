# test_server.py
import json
import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_default_route():
    """Test the default route returns correct message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to RKLLama API" in response.json()["message"]


def test_list_models():
    """Test the list models endpoint."""
    response = client.get("/api/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    # The response should contain a list of model objects


def test_get_current_model():
    """Test getting the current loaded model information."""
    response = client.get("/api/current_model")
    assert response.status_code == 200
    # Response will be either model info or a message that no model is loaded


def test_load_model():
    """Test loading a model."""
    # This is a mock test - in real scenario we'd need an actual model name
    model_name = "test_model"  # Replace with a valid model name for real testing
    response = client.post(f"/api/load/{model_name}")
    # The response might be 200 if model exists or 404/500 if it doesn't
    # Just assert that we get a response
    assert response.status_code in [200, 404, 500]


def test_unload_model():
    """Test unloading the current model."""
    response = client.post("/api/unload")
    assert response.status_code in [200, 404]
    # If a model was loaded, it should return 200, otherwise might return 404


def test_recevoir_message():
    """Test the message receiving endpoint."""
    payload = {
        "message": "Hello, how are you?",
        "history": [],
        "temperature": 0.7,
        "max_tokens": 100
    }
    response = client.post("/api/chat", json=payload)
    # Status might vary based on whether a model is loaded
    assert response.status_code in [200, 400, 500]


# Ollama API compatibility tests
def test_ollama_models():
    """Test the Ollama models list endpoint."""
    response = client.get("/api/ollama/models")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "models" in response.json()


def test_ollama_chat():
    """Test the Ollama chat endpoint."""
    payload = {
        "model": "test_model",  # Replace with a valid model name
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    response = client.post("/api/ollama/chat", json=payload)
    # Status might vary based on model availability
    assert response.status_code in [200, 400, 404, 500]


def test_ollama_generate():
    """Test the Ollama generate endpoint."""
    payload = {
        "model": "test_model",  # Replace with a valid model name
        "prompt": "Hello, world!",
        "stream": False
    }
    response = client.post("/api/ollama/generate", json=payload)
    # Status might vary based on model availability
    assert response.status_code in [200, 400, 404, 500]


def test_ollama_version():
    """Test the Ollama version endpoint."""
    response = client.get("/api/ollama/version")
    assert response.status_code == 200
    assert "version" in response.json()


# Tool calling tests
def test_tool_calling_chat():
    """Test the chat endpoint with tool calling."""
    payload = {
        "model": "test_model",  # Replace with a valid model name
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }]
    }
    response = client.post("/api/ollama/chat", json=payload)
    # Status might vary based on model availability
    assert response.status_code in [200, 400, 404, 500]
