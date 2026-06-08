import pytest
import datetime
from unittest.mock import MagicMock, patch

from core.model.Model import Model
from core.model.ModelPath import ModelPath
from core.model.ModelFile import ModelFile
from core.model.ModelConfig import FullModelParameters


@pytest.fixture
def test_model():
    from core.model.ModelInfo import ModelInfo, ModelDetails
    from core.model.ModelType import ModelType

    details = ModelDetails(
        parameter_size="3B",
        quantization_level="Q4_K_M",
        model_format="rkllm",
        model_family="qwen2",
        model_families=["qwen2"],
    )

    model_info = ModelInfo(
        name="test-model",
        model="test-model",
        created_at_dt=datetime.datetime.now(),
        modified_at_dt=datetime.datetime.now(),
        size=12345,
        digest="abcdef",
        details=details,
        model_type=ModelType.RKLLM,
    )

    return Model(
        id="test-model",
        st_atime=0.0,
        st_mtime=0.0,
        st_ctime=0.0,
        size=12345,
        digest="abcdef",
        model_path=ModelPath(
            model_name="test-model",
            endpoint_model_file="test-model.rkllm",
            endpoint_model_file_size=12345,
        ),
        model_info=model_info,
    )


def test_list_models_rkllama(api_client, test_model):
    with patch("core.model.Model.Model.list", return_value=[test_model]):
        response = api_client.get("/models")
        assert response.status_code == 200
        assert "models" in response.json()


def test_get_model_rkllama(api_client, test_model):
    mock_model_path = ModelPath(
        model_name="test-model",
        endpoint_model_file="test-model.rkllm",
        endpoint_model_file_size=12345,
    )

    with (
        patch(
            "api.routes.rkllama.ModelPath.from_model_id", return_value=mock_model_path
        ),
        patch("api.routes.rkllama.Model.load", return_value=test_model),
    ):
        response = api_client.get("/models/test-model")
        assert response.status_code == 200


def test_load_unload_model_rkllama(api_client, test_model):
    mock_model_file = MagicMock(spec=ModelFile)
    mock_model_file.model_name = "test-model"
    mock_model_file.model = test_model
    mock_model_file.full_model_parameters = MagicMock(spec=FullModelParameters)

    mock_worker = MagicMock()
    mock_process = MagicMock()

    mock_wm = MagicMock()
    mock_wm.add_worker.return_value = (mock_worker, mock_process)
    mock_wm.workers = {"test-model": mock_worker}

    mock_worker_model_info = MagicMock()
    mock_worker_model_info.size = 12345
    mock_worker_model_info.expires_at = datetime.datetime.now()
    mock_worker_model_info.loaded_at = datetime.datetime.now()
    mock_base_domain_id = MagicMock()
    mock_base_domain_id.value = 1
    mock_worker_model_info.base_domain_id = mock_base_domain_id
    mock_worker_model_info.last_call = datetime.datetime.now()
    mock_worker.worker_model_info = mock_worker_model_info

    mock_model_path = ModelPath(
        model_name="test-model",
        endpoint_model_file="test-model.rkllm",
        endpoint_model_file_size=12345,
    )

    with (
        patch(
            "core.model.ModelPath.ModelPath.from_model_id", return_value=mock_model_path
        ),
        patch("core.model.ModelFile.ModelFile.load", return_value=mock_model_file),
        patch("core.processing.WorkerManager.get_worker_manager", return_value=mock_wm),
        patch("core.processing.WorkerManager.worker_managers", [mock_wm]),
    ):
        response = api_client.post("/load_model", json={"model_name": "test-model"})
        assert (
            response.status_code == 200
        ), f"Response: {response.status_code} - {response.text}"
        assert "loaded successfully" in response.json()["message"]

        # Test Current models
        response = api_client.get("/current_models")
        assert response.status_code == 200
        assert "models" in response.json()
        assert len(response.json()["models"]) == 1
        assert response.json()["models"][0]["name"] == "test-model"

        # Test Unload model
        response = api_client.post("/unload_model")
        assert response.status_code == 200
        assert "unloaded" in response.json()["message"]


def test_list_tags_ollama(api_client, test_model):
    with patch("core.model.Model.Model.list", return_value=[test_model]):
        response = api_client.get("/api/tags")
        assert response.status_code == 200
        assert "models" in response.json()
