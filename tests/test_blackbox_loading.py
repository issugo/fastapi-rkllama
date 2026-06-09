import os
import json
import shutil
import pytest
from fastapi.testclient import TestClient
from app.main import app
from core.config.config_utils import get_path


@pytest.fixture
def dummy_model():
    models_dir = get_path("models")
    model_name = "qwen3-test"
    model_file = "Qwen3-4B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm"
    model_id = f"{model_name}/{model_file}"

    full_dir = os.path.join(models_dir, model_name)
    os.makedirs(full_dir, exist_ok=True)

    # 1. Create dummy .rkllm file
    file_path = os.path.join(full_dir, model_file)
    with open(file_path, "wb") as f:
        f.write(b"dummy model content")

    # 1b. Create dummy Modelfile
    modelfile_path = os.path.join(full_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(f"FROM {model_file}\n")

    # 2. Create dummy .metadata file (SimpleModelMetadata format)
    metadata_path = f"{file_path}.metadata"
    metadata_content = {
        "name": "qwen3",
        "architecture": "qwen2",
        "quantization": "w8a8",
        "parameters": 4000000000,
        "context_length": 16384,
        "temperature": 0.7,
        "model_type": "RKLLM",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata_content, f)

    # 3. Create dummy .HfFileInfo file
    hf_file_info_path = f"{file_path}.HfFileInfo"
    hf_file_info_content = {
        "name": model_id,
        "size": 100,
        "type": "file",
        "blob_id": "dummy-blob",
        "lfs": {"size": 100, "sha256": "dummy-sha256", "pointer_size": 100},
        "last_commit": None,
        "security": None,
    }
    with open(hf_file_info_path, "w") as f:
        json.dump(hf_file_info_content, f)

    # 4. Create dummy ModelInfo.json file
    hf_model_info_path = os.path.join(full_dir, "ModelInfo.json")
    hf_model_info_content = {
        "id": model_id,
        "private": False,
        "tags": ["qwen2", "rkllm"],
        "downloads": 10,
        "likes": 5,
        "modelId": model_id,
        "author": "qwen3-test",
        "sha": "dummy-sha",
        "lastModified": "2024-01-01T00:00:00.000Z",
        "gated": False,
        "disabled": False,
        "config": {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "tokenizer_config": {},
        },
        "cardData": {
            "base_model": ["qwen/qwen2"],
            "tags": ["rkllm"],
            "params": 4000000000,
        },
        "siblings": [],
        "createdAt": "2024-01-01T00:00:00.000Z",
        "usedStorage": 100,
        "languages": ["en"],
    }
    with open(hf_model_info_path, "w") as f:
        json.dump(hf_model_info_content, f)

    # 5. Create dummy .config file
    config_path = f"{file_path}.config"
    with open(config_path, "w") as f:
        f.write(f"# FROM={model_file}\n")
        f.write("# HUGGINGFACE_PATH=qwen3-test\n")
        f.write("{}\n")

    yield model_id

    # Cleanup
    shutil.rmtree(full_dir, ignore_errors=True)


def test_model_loading_unloading_blackbox(dummy_model):
    """
    Blackbox test for model loading and unloading using FastAPI TestClient.
    This test verifies that:
    1. A model can be loaded via POST /load_model.
    2. The loaded model appears in GET /current_models.
    3. The model can be unloaded via POST /unload_model.
    4. The model no longer appears in GET /current_models after unloading.
    """
    model_id = dummy_model

    # Use FastAPI TestClient for blackbox testing of the app
    # This simulates external HTTP requests to the FastAPI application
    with TestClient(app) as client:
        # 1. Load the model
        load_payload = {"model_name": model_id}
        response = client.post("/load_model", json=load_payload)
        assert response.status_code == 200, f"Load failed: {response.text}"
        assert "loaded successfully" in response.json().get("message", "")

        # 2. Verify model is loaded via /current_models
        response = client.get("/current_models")
        assert response.status_code == 200
        current_models = response.json().get("models", [])
        assert any(
            m["name"] == model_id for m in current_models
        ), f"Model {model_id} not found in current_models"

        # 3. Unload the model
        response = client.post("/unload_model")
        assert response.status_code == 200, f"Unload failed: {response.text}"
        assert "successfully unloaded" in response.json().get("message", "")

        # 4. Verify model is no longer in /current_models
        response = client.get("/current_models")
        assert response.status_code == 200
        current_models = response.json().get("models", [])
        assert not any(
            m["name"] == model_id for m in current_models
        ), f"Model {model_id} still found in current_models after unload"
