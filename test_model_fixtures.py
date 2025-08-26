# test_fixtures.py
import pytest
import os
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="module")
def temp_models_dir():
    """Create a temporary directory for test models."""
    original_models_dir = os.environ.get("MODELS_DIR", "~/RKLLAMA/models")
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    # Store the original value
    os.environ["MODELS_DIR"] = temp_dir

    # Create a simple mock model structure for testing
    model_dir = Path(temp_dir) / "test_model"
    model_dir.mkdir(exist_ok=True)

    # Create a mock Modelfile
    with open(model_dir / "Modelfile", "w") as f:
        f.write('FROM="test_model.rkllm"\nHUGGINGFACE_PATH="test/repo"\nSYSTEM="Test system prompt"\n')

    # Create an empty .rkllm file (just for structure testing)
    with open(model_dir / "test_model.rkllm", "w") as f:
        f.write("Mock model file")

    yield temp_dir

    # Cleanup and restore original value
    shutil.rmtree(temp_dir)
    os.environ["MODELS_DIR"] = original_models_dir


@pytest.fixture
def mock_model_loaded(monkeypatch):
    """Mock a loaded model for testing."""

    def mock_load(*args, **kwargs):
        return True

    def mock_get_current():
        return {
            "name": "test_model",
            "file": "test_model.rkllm",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100
            }
        }

    monkeypatch.setattr("server.load_model", mock_load)
    monkeypatch.setattr("server.get_current_model", mock_get_current)
    monkeypatch.setattr("server.current_model", "test_model")

    yield

    # Reset the mock
    monkeypatch.undo()
