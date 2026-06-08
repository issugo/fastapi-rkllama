import pytest
from unittest.mock import MagicMock, patch

from core.model.ModelConfig import FullModelParameters
from core.model.ModelFile import ModelFile
from core.model.ModelPath import ModelPath
from core.model.Model import Model
from core.model.ModelType import ModelType
from core.processing.WorkerManager import WorkerManager


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
        mirostat_eta=0.0,
    )


@pytest.fixture
def mock_model_file(base_full_params):
    mock_file = MagicMock(spec=ModelFile)
    mock_file.model_id = "test-hf-model"
    mock_file.model_name = "test-hf-model"

    mock_model = MagicMock(spec=Model)
    mock_model.model_type = ModelType.RKLLM
    mock_file.model = mock_model

    mock_file.full_model_parameters = base_full_params
    return mock_file


def test_hf_load_model_with_custom_temperature_and_ctx(api_client, mock_model_file):
    mock_model_path = MagicMock(spec=ModelPath)

    mock_wm = MagicMock(spec=WorkerManager)
    mock_worker = MagicMock()
    mock_process = MagicMock()
    mock_wm.add_worker.return_value = (mock_worker, mock_process)

    custom_temp = 0.3
    custom_ctx = 2048
    custom_num_predict = 100

    payload = {
        "model_name": "test-hf-model",
        "temperature": custom_temp,
        "num_ctx": custom_ctx,
        "num_predict": custom_num_predict,
    }

    with (
        patch(
            "core.model.ModelPath.ModelPath.from_model_id", return_value=mock_model_path
        ),
        patch("core.model.ModelFile.ModelFile.load", return_value=mock_model_file),
        patch("core.processing.WorkerManager.get_worker_manager", return_value=mock_wm),
    ):
        response = api_client.post("/load_model", json=payload)

        assert response.status_code == 200

        # Verify that add_worker was called with options overridden correctly
        mock_wm.add_worker.assert_called_once()
        called_kwargs = mock_wm.add_worker.call_args[1]
        called_params = called_kwargs.get("full_model_parameters")

        assert called_params is not None
        assert called_params.temperature == custom_temp
        assert called_params.num_ctx == custom_ctx
        assert called_params.max_new_tokens == custom_num_predict


def test_hf_load_model_with_nested_options(api_client, mock_model_file):
    mock_model_path = MagicMock(spec=ModelPath)

    mock_wm = MagicMock(spec=WorkerManager)
    mock_worker = MagicMock()
    mock_process = MagicMock()
    mock_wm.add_worker.return_value = (mock_worker, mock_process)

    payload = {
        "model_name": "test-hf-model",
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "repeat_penalty": 1.4,
            "mirostat": True,
        },
    }

    with (
        patch(
            "core.model.ModelPath.ModelPath.from_model_id", return_value=mock_model_path
        ),
        patch("core.model.ModelFile.ModelFile.load", return_value=mock_model_file),
        patch("core.processing.WorkerManager.get_worker_manager", return_value=mock_wm),
    ):
        response = api_client.post("/load_model", json=payload)

        assert (
            response.status_code == 200
        ), f"Response: {response.status_code} - {response.text}"

        # Verify overridden parameters
        mock_wm.add_worker.assert_called_once()
        called_kwargs = mock_wm.add_worker.call_args[1]
        called_params = called_kwargs.get("full_model_parameters")

        assert called_params is not None
        assert called_params.temperature == 0.1
        assert called_params.num_ctx == 8192
        assert called_params.repeat_penalty == 1.4
        assert called_params.mirostat is True


def test_hf_load_model_with_default_options(
    api_client, mock_model_file, base_full_params
):
    mock_model_path = MagicMock(spec=ModelPath)

    mock_wm = MagicMock(spec=WorkerManager)
    mock_worker = MagicMock()
    mock_process = MagicMock()
    mock_wm.add_worker.return_value = (mock_worker, mock_process)

    payload = {"model_name": "test-hf-model"}

    with (
        patch(
            "core.model.ModelPath.ModelPath.from_model_id", return_value=mock_model_path
        ),
        patch("core.model.ModelFile.ModelFile.load", return_value=mock_model_file),
        patch("core.processing.WorkerManager.get_worker_manager", return_value=mock_wm),
    ):
        response = api_client.post("/load_model", json=payload)

        assert response.status_code == 200

        # Verify default parameters are used
        mock_wm.add_worker.assert_called_once()
        called_kwargs = mock_wm.add_worker.call_args[1]
        called_params = called_kwargs.get("full_model_parameters")

        assert called_params.temperature == base_full_params.temperature
        assert called_params.num_ctx == base_full_params.num_ctx
        assert called_params.max_new_tokens == base_full_params.max_new_tokens
