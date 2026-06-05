import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from core.model.ModelConfig import FullModelParameters
from core.model.ModelFile import ModelFile
from core.model.ModelPath import ModelPath
from core.model.Model import Model
from core.model.ModelType import ModelType
from core.backends.backend import BackendType
from core.processing.WorkerManager import WorkerManager
from core.api.parameters.ollama_responses import OllamaGenerateResponse

@pytest.fixture
def base_full_params():
    return FullModelParameters(
        num_ctx=4096,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.7,
        seed=42,
        stop="AI assistant:",
        num_predict=42,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        enable_thinking=False,
        max_new_tokens=2048,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        mirostat=False,
        mirostat_tau=0.0,
        mirostat_eta=0.0
    )

@pytest.fixture
def mock_model_file(base_full_params):
    mock_file = MagicMock(spec=ModelFile)
    mock_file.model_id = "test-qwen2"
    mock_file.model_name = "test-qwen2"
    mock_file.SYSTEM = "You are a helpful assistant."
    
    mock_model = MagicMock(spec=Model)
    mock_model.model_type = ModelType.RKLLM
    mock_file.model = mock_model
    
    mock_file.full_model_parameters = base_full_params
    return mock_file

def test_ollama_generate_with_custom_temperature_and_ctx(api_client, mock_model_file):
    # Mocking necessary core parts to avoid loading actual models from disk
    mock_model_path = MagicMock(spec=ModelPath)
    
    mock_wm = MagicMock(spec=WorkerManager)
    mock_worker = MagicMock()
    mock_process = MagicMock()
    mock_wm.add_worker.return_value = (mock_worker, mock_process)

    # Specific configuration to load model with
    custom_temp = 0.3
    custom_ctx = 2048
    custom_num_predict = 100

    payload = {
        "model": "test-qwen2",
        "prompt": "Hello world",
        "options": {
            "temperature": custom_temp,
            "num_ctx": custom_ctx,
            "num_predict": custom_num_predict
        },
        "stream": False
    }

    with patch("api.routes.ollama_new.ModelPath.from_model_id", return_value=mock_model_path), \
         patch("api.routes.ollama_new.ModelFile.load", return_value=mock_model_file), \
         patch("api.routes.ollama_new.get_worker_manager", return_value=mock_wm), \
         patch("api.routes.ollama_new.GenerateEndpointHandler.handle_request") as mock_handle_request:
        
        mock_handle_request.return_value = OllamaGenerateResponse(
            model="test-qwen2",
            created_at="2026-06-03T12:00:00Z",
            response="Mocked response",
            done=True
        )

        response = api_client.post("/api/generate", json=payload)
        
        assert response.status_code == 200
        
        # Verify that add_worker was called with options overridden correctly
        mock_wm.add_worker.assert_called_once()
        called_kwargs = mock_wm.add_worker.call_args[1]
        called_params = called_kwargs.get("full_model_parameters")
        
        assert called_params is not None
        assert called_params.temperature == custom_temp
        assert called_params.num_ctx == custom_ctx
        assert called_params.max_new_tokens == custom_num_predict

        # Verify that handle_request was called with the overridden parameters
        mock_handle_request.assert_called_once()
        handler_kwargs = mock_handle_request.call_args[1]
        assert handler_kwargs.get("options").temperature == custom_temp
        assert handler_kwargs.get("options").num_ctx == custom_ctx
        assert handler_kwargs.get("options").max_new_tokens == custom_num_predict


def test_ollama_generate_with_default_options(api_client, mock_model_file, base_full_params):
    mock_model_path = MagicMock(spec=ModelPath)
    
    mock_wm = MagicMock(spec=WorkerManager)
    mock_worker = MagicMock()
    mock_process = MagicMock()
    mock_wm.add_worker.return_value = (mock_worker, mock_process)

    payload = {
        "model": "test-qwen2",
        "prompt": "Hello world",
        "stream": False
    }

    with patch("api.routes.ollama_new.ModelPath.from_model_id", return_value=mock_model_path), \
         patch("api.routes.ollama_new.ModelFile.load", return_value=mock_model_file), \
         patch("api.routes.ollama_new.get_worker_manager", return_value=mock_wm), \
         patch("api.routes.ollama_new.GenerateEndpointHandler.handle_request") as mock_handle_request:
        
        mock_handle_request.return_value = OllamaGenerateResponse(
            model="test-qwen2",
            created_at="2026-06-03T12:00:00Z",
            response="Mocked response",
            done=True
        )

        response = api_client.post("/api/generate", json=payload)
        
        assert response.status_code == 200
        
        # Verify default parameters are used
        mock_wm.add_worker.assert_called_once()
        called_kwargs = mock_wm.add_worker.call_args[1]
        called_params = called_kwargs.get("full_model_parameters")
        
        assert called_params.temperature == base_full_params.temperature
        assert called_params.num_ctx == base_full_params.num_ctx
        assert called_params.max_new_tokens == base_full_params.max_new_tokens


def test_ollama_generate_with_all_overridden_options(api_client, mock_model_file):
    mock_model_path = MagicMock(spec=ModelPath)
    
    mock_wm = MagicMock(spec=WorkerManager)
    mock_worker = MagicMock()
    mock_process = MagicMock()
    mock_wm.add_worker.return_value = (mock_worker, mock_process)

    payload = {
        "model": "test-qwen2",
        "prompt": "Hello world",
        "options": {
            "temperature": 0.2,
            "num_ctx": 1024,
            "num_predict": 150,
            "seed": 999,
            "top_k": 20,
            "top_p": 0.5,
            "repeat_penalty": 1.5,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.8,
            "mirostat": 1,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1
        },
        "stream": False
    }

    with patch("api.routes.ollama_new.ModelPath.from_model_id", return_value=mock_model_path), \
         patch("api.routes.ollama_new.ModelFile.load", return_value=mock_model_file), \
         patch("api.routes.ollama_new.get_worker_manager", return_value=mock_wm), \
         patch("api.routes.ollama_new.GenerateEndpointHandler.handle_request") as mock_handle_request:
        
        mock_handle_request.return_value = OllamaGenerateResponse(
            model="test-qwen2",
            created_at="2026-06-03T12:00:00Z",
            response="Mocked response",
            done=True
        )

        response = api_client.post("/api/generate", json=payload)
        
        assert response.status_code == 200
        
        # Verify overridden parameters are all loaded correctly in WorkerManager.add_worker
        mock_wm.add_worker.assert_called_once()
        called_kwargs = mock_wm.add_worker.call_args[1]
        called_params = called_kwargs.get("full_model_parameters")
        
        assert called_params.temperature == 0.2
        assert called_params.num_ctx == 1024
        assert called_params.max_new_tokens == 150
        assert called_params.seed == 999
        assert called_params.top_k == 20
        assert called_params.top_p == 0.5
        assert called_params.repeat_penalty == 1.5
        assert called_params.presence_penalty == 0.5
        assert called_params.frequency_penalty == 0.8
        assert called_params.mirostat == 1
        assert called_params.mirostat_tau == 5.0
        assert called_params.mirostat_eta == 0.1

