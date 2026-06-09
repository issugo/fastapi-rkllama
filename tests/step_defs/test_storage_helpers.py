"""
Step definitions for testing the storage_helpers package.

This module contains BDD step definitions to verify HuggingFace and Ollama
file system helpers, pull suppliers, and storage helpers. It isolates the tests
from remote networks by mocking HTTP calls and local file storage paths.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Any
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from unittest.mock import MagicMock, patch

from core.model.storage_helpers.HuggingfaceFileSystem import HuggingfaceFileSystem
from core.model.storage_helpers.OllamaFileSystem import OllamaFileSystem
from core.model.storage_helpers.OllamaPullSupplier import OllamaPullSupplier
from core.model.storage_helpers.RKPullSupplier import RKPullSupplier
from core.model.storage_helpers.model_pull import pull_model
from core.model.storage_helpers.SupplierFileInfo import Supplier
from core.model.ModelType import ModelType
from core.config.config_utils import get_settings

# Link to feature file
scenarios("../features/storage_helpers.feature")

# --- Mock Manifest Data for Ollama Pull ---
manifest_mock = {
    "schemaVersion": 2,
    "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
    "config": {
        "mediaType": "application/vnd.docker.container.image.v1+json",
        "digest": "sha256:cfg12345",
        "size": 100,
    },
    "layers": [
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:model12345",
            "size": 12,
        },
        {
            "mediaType": "application/vnd.ollama.image.system",
            "digest": "sha256:system12345",
            "size": 10,
        },
        {
            "mediaType": "application/vnd.ollama.image.template",
            "digest": "sha256:template12345",
            "size": 10,
        },
        {
            "mediaType": "application/vnd.ollama.image.license",
            "digest": "sha256:license12345",
            "size": 10,
        },
    ],
}


# --- Test Suppliers subclassing abstract PullSupplier ---
class TestRKPullSupplier(RKPullSupplier):
    """
    Subclass of RKPullSupplier for testing purposes.
    """

    def __init__(self, model_name: str, file: str, repo: str, logger: Any):
        self._model_name = model_name
        self._file = file
        self._repo = repo
        self._logger = logger

    @property
    def logger(self) -> Any:
        return self._logger

    def check_params(self) -> Any | None:
        return None

    def model_data(self) -> Tuple[str, str, str | None, Supplier]:
        return self._model_name, self._file, self._repo, Supplier.HUGGINGFACE

    def model_type(
        self, model_name: str, file: str, repo: str
    ) -> Tuple[ModelType | None, Any]:
        return ModelType.RKLLM, None


class TestOllamaPullSupplier(OllamaPullSupplier):
    """
    Subclass of OllamaPullSupplier for testing purposes.
    """

    def __init__(self, model_name: str, file: str, repo: str, logger: Any):
        self._model_name = model_name
        self._file = file  # tag
        self._repo = repo
        self._logger = logger

    @property
    def logger(self) -> Any:
        return self._logger

    def check_params(self) -> Any | None:
        return None

    def model_data(self) -> Tuple[str, str, str | None, Supplier]:
        return self._model_name, self._file, self._repo, Supplier.OLLAMA

    def model_type(
        self, model_name: str, file: str, repo: str
    ) -> Tuple[ModelType | None, Any]:
        return ModelType.GGUF, None


# --- Shared Test Context Fixture ---
@pytest.fixture
def test_context():
    """
    Shared test context to store values, results, errors and mocks between steps.
    """
    return {}


@pytest.fixture
def temp_models_dir():
    """
    Creates a temporary models directory and cleans it up after the test.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def patch_settings(temp_models_dir):
    """
    Patches settings.paths.models to point to the temporary models directory.
    """
    settings = get_settings()
    original_models_path = settings.paths.models
    settings.paths.models = temp_models_dir
    yield settings
    settings.paths.models = original_models_path


# --- Scenario 1 Steps ---
@given("a HuggingFace path validator is available")
def hf_validator_available(test_context):
    pass


@when(parsers.parse('I validate the HuggingFace path "{path}"'))
def validate_hf_path(test_context, path):
    try:
        res = HuggingfaceFileSystem.validate_huggingface_path(path)
        test_context["result"] = res
        test_context["error"] = None
    except ValueError as e:
        test_context["result"] = None
        test_context["error"] = e


@when(parsers.parse('I validate the HuggingFace path "{path}" with author "{author}"'))
def validate_hf_path_with_author(test_context, path, author):
    try:
        res = HuggingfaceFileSystem.validate_huggingface_path(path, author=author)
        test_context["result"] = res
        test_context["error"] = None
    except ValueError as e:
        test_context["result"] = None
        test_context["error"] = e


@then(parsers.parse('the path should be valid and return "{expected}"'))
def check_hf_path_valid(test_context, expected):
    assert test_context.get("error") is None
    assert test_context.get("result") == expected


@then("it should raise a ValueError")
def check_value_error(test_context):
    assert test_context.get("error") is not None
    assert isinstance(test_context.get("error"), ValueError)


# --- Scenario 2 Steps ---
@given("an Ollama file system utility is available")
def ollama_fs_available(test_context):
    pass


@when(
    parsers.parse(
        'I request the model path for "{model_name}" with api flag set to {api_flag}'
    )
)
def get_model_path(test_context, model_name, api_flag):
    api_bool = api_flag.lower() == "true"
    res = OllamaFileSystem.model_path(model_name, api=api_bool)
    test_context["result"] = res


@then(parsers.parse('the model path should end with "{expected}"'))
def check_model_path(test_context, expected):
    assert test_context.get("result").endswith(expected)


@when(
    parsers.parse(
        'I request the blob URL for digest "{digest}" and model "{model_name}"'
    )
)
def get_blob_url(test_context, digest, model_name):
    try:
        res = OllamaFileSystem.blob_url(digest, model_name)
        test_context["result"] = res
        test_context["error"] = None
    except ValueError as e:
        test_context["result"] = None
        test_context["error"] = e


@then(parsers.parse('the blob URL should end with "{expected}"'))
def check_blob_url(test_context, expected):
    assert test_context.get("error") is None
    assert test_context.get("result").endswith(expected)


@when(
    parsers.parse(
        'I request the model URL for digest "{digest}" and model "{model_name}"'
    )
)
def get_model_url(test_context, digest, model_name):
    res = OllamaFileSystem.model_url(digest, model_name)
    test_context["result"] = res


@then(parsers.parse('the model URL should end with "{expected}"'))
def check_model_url(test_context, expected):
    assert test_context.get("result").endswith(expected)


@when("I request the blob URL with an empty digest")
def get_blob_url_empty(test_context):
    try:
        OllamaFileSystem.blob_url("", "llama3")
        test_context["error"] = None
    except ValueError as e:
        test_context["error"] = e


# --- Scenario 3 Steps ---
@given(parsers.parse('a mocked HuggingFace API returning model metadata for "{repo}"'))
def mock_hf_api(test_context, repo):
    test_context["repo"] = repo
    test_context["description"] = ""
    test_context["metadata"] = {
        "tags": [],
        "cardData": {},
        "license": "apache-2.0",
        "siblings": [{"rfilename": "model.rkllm"}, {"rfilename": "LICENSE"}],
        "sibling_models": [{"rfilename": "model.rkllm"}],
    }


@given(
    parsers.parse(
        'the model description contains architecture "{arch}" and quantization "{quant}"'
    )
)
def set_hf_desc(test_context, arch, quant):
    test_context["description"] = (
        f"7B Model with architecture {arch} and quantization {quant}."
    )
    test_context["metadata"]["description"] = test_context["description"]


@when(parsers.parse('I load the model info for HuggingFace path "{path}"'))
def load_hf_model_info(test_context, path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = test_context["metadata"]

    with patch(
        "core.model.storage_helpers.HuggingfaceFileSystem.requests.get",
        return_value=mock_resp,
    ):
        res = HuggingfaceFileSystem.load_model_info(path)
        test_context["result"] = res


@then(parsers.parse('the response should contain the architecture "{arch}"'))
def check_architecture(test_context, arch):
    assert test_context.get("result")["architecture"] == arch


@then(parsers.parse('the response should contain the quantization "{quant}"'))
def check_quantization(test_context, quant):
    assert test_context.get("result")["quantization"] == quant


@then("the response should contain English language in tags")
def check_english_lang(test_context):
    assert "en" in test_context.get("result")["languages"]


@then("the response should contain the license name and URL")
def check_license(test_context):
    assert test_context.get("result")["license_name"] is not None
    assert "license_url" in test_context.get("result")


# --- Scenario 4 Steps ---
@given(
    parsers.parse(
        'a mocked Ollama registry config endpoint for model "{model}" and digest "{digest}"'
    )
)
def mock_ollama_config(test_context, model, digest):
    test_context["model"] = model
    test_context["digest"] = digest
    test_context["config_json"] = {
        "architecture": "amd64",
        "os": "linux",
        "created": "2026-06-03T12:00:00Z",
        "description": f"{model}:latest",
        "config": {"Labels": {"description": "mocked description"}},
    }


@when(
    parsers.parse(
        'I load the Ollama config for digest "{digest}", model "{model}", and tag "{tag}"'
    )
)
def load_ollama_config(test_context, digest, model, tag):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = test_context["config_json"]

    with patch(
        "core.model.storage_helpers.OllamaFileSystem.requests.get",
        return_value=mock_resp,
    ):
        cfg, info = OllamaFileSystem.load_config(digest, model, tag)
        test_context["cfg"] = cfg
        test_context["info"] = info


@then(
    parsers.parse('the returned configuration should contain the architecture "{arch}"')
)
def check_ollama_config_arch(test_context, arch):
    assert test_context.get("cfg")["architecture"] == arch


@then(parsers.parse('the info dictionary should have description "{desc}"'))
def check_ollama_info_desc(test_context, desc):
    assert test_context.get("info")["description"] == desc


# --- Scenarios 5, 6, 7 Shared Setup Steps ---
@given("the fastapi-rkllama storage directories are prepared")
def prepare_storage(test_context, temp_models_dir, patch_settings):
    test_context["models_dir"] = temp_models_dir
    test_context["settings"] = patch_settings


@given("a mocked HuggingFace HfFileSystem and download registry")
def mock_hf_success(test_context):
    mock_fs = MagicMock()
    mock_fs.info.return_value = {
        "name": "my-author/qwen2-7b/qwen2-7b-rk3588-w8a8.rkllm",
        "size": 12,
        "type": "file",
        "blob_id": "dummy-blob",
        "lfs": {
            "size": 12,
            "sha256": "27ae60300386eb5e825976dd3346bd6329e0effb7948961e10c90aefc003874c",
            "pointer_size": 100,
        },
        "last_commit": None,
        "security": None,
    }

    mock_hf_info = {
        "id": "my-author/qwen2-7b",
        "private": False,
        "tags": ["rkllm"],
        "downloads": 100,
        "likes": 10,
        "modelId": "my-author/qwen2-7b",
        "author": "my-author",
        "sha": "dummy-sha",
        "lastModified": "2026-06-03T12:00:00.000Z",
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
            "params": 7000000000,
        },
        "siblings": [
            {"rfilename": "qwen2-7b-rk3588-w8a8.rkllm"},
            {"rfilename": "LICENSE"},
        ],
        "createdAt": "2026-06-03T12:00:00.000Z",
        "usedStorage": 100,
        "languages": ["en"],
        "description": "Model with architecture qwen2 and quantization w8a8.",
    }
    test_context["fs"] = mock_fs
    test_context["hf_info"] = mock_hf_info


@given("a mocked HuggingFace registry that fails during file download")
def mock_hf_fail(test_context):
    mock_fs = MagicMock()
    mock_fs.info.return_value = {
        "name": "my-author/qwen2-7b/qwen2-7b-rk3588-w8a8-fail.rkllm",
        "size": 1000,
        "type": "file",
        "blob_id": "dummy-blob",
        "lfs": {"size": 1000, "sha256": "fail-sha256", "pointer_size": 100},
        "last_commit": None,
        "security": None,
    }

    mock_hf_info = {
        "id": "my-author/qwen2-7b",
        "private": False,
        "tags": ["rkllm"],
        "downloads": 100,
        "likes": 10,
        "modelId": "my-author/qwen2-7b",
        "author": "my-author",
        "sha": "dummy-sha",
        "lastModified": "2026-06-03T12:00:00.000Z",
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
            "params": 7000000000,
        },
        "siblings": [{"rfilename": "qwen2-7b-rk3588-w8a8-fail.rkllm"}],
        "createdAt": "2026-06-03T12:00:00.000Z",
        "usedStorage": 1000,
        "languages": ["en"],
        "description": "Model with architecture qwen2 and quantization w8a8.",
    }
    test_context["fs"] = mock_fs
    test_context["hf_info"] = mock_hf_info


@when(parsers.parse('I pull the HuggingFace model "{repo}" with file "{file}"'))
def pull_hf_model(test_context, repo, file):
    model_name = repo.split("/")[-1]
    logger = MagicMock()
    supplier = TestRKPullSupplier(
        model_name=model_name, file=file, repo=repo, logger=logger
    )

    def mock_get(url, *args, **kwargs):
        if "fail.rkllm" in url or "fail-sha256" in url:
            resp = MagicMock()
            resp.status_code = 200
            resp.iter_content.side_effect = ConnectionError(
                "Simulated download failure"
            )
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = None
            return resp

        resp = MagicMock()
        resp.status_code = 200
        if "LICENSE" in url:
            resp.content = b"Mocked Apache License"
        else:
            resp.iter_content.return_value = [b"chunk1", b"chunk2"]
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    mock_request = MagicMock()
    mock_request.url = "http://localhost/pull"

    with (
        patch(
            "core.model.storage_helpers.RKPullSupplier.HfFileSystem",
            return_value=test_context["fs"],
        ),
        patch(
            "core.model.storage_helpers.RKPullSupplier.HuggingfaceFileSystem.load_model_info",
            return_value=test_context["hf_info"],
        ),
        patch(
            "core.model.storage_helpers.RKPullSupplier.requests.get",
            side_effect=mock_get,
        ),
        patch(
            "core.model.storage_helpers.model_pull.requests.get", side_effect=mock_get
        ),
    ):
        res = pull_model(request=mock_request, pull_supplier=supplier)
        test_context["result"] = res
        test_context["supplier"] = supplier
        test_context["model_name"] = model_name
        test_context["file"] = file


@then("the pull operation should succeed")
def check_pull_success(test_context):
    assert not str(test_context.get("result")).startswith(
        "Error"
    ), f"Pull failed with error: {test_context.get('result')}"


@then("the RKLLM model file and metadata should be correctly saved to disk")
def check_rkllm_saved(test_context):
    models_dir = Path(test_context.get("models_dir"))
    model_name = test_context.get("model_name")
    file = test_context.get("file")

    model_dir = models_dir / model_name
    assert model_dir.exists()

    model_info_json = model_dir / "ModelInfo.json"
    assert model_info_json.exists()

    hf_file_info_path = model_dir / f"{file}.HfFileInfo"
    assert hf_file_info_path.exists()

    model_link = model_dir / file
    assert model_link.exists()
    assert model_link.is_symlink()


@then("the model should be unlocked")
def check_model_unlocked(test_context):
    models_dir = Path(test_context.get("models_dir"))
    model_name = test_context.get("model_name")
    lock_file = models_dir / model_name / "lock"
    assert not lock_file.exists()


# --- Scenario 6 Steps ---
@given("a mocked Ollama registry API for pulling")
def mock_ollama_pull(test_context):
    pass


@when(parsers.parse('I pull the Ollama model "{repo}" with tag "{tag}"'))
def pull_ollama_model(test_context, repo, tag):
    model_name = repo
    logger = MagicMock()
    supplier = TestOllamaPullSupplier(
        model_name=model_name, file=tag, repo=repo, logger=logger
    )

    def mock_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200

        if "manifests/" in url:
            resp.json.return_value = manifest_mock
        elif "cfg12345" in url:
            resp.json.return_value = {
                "model_format": "gguf",
                "model_family": "qwen2",
                "model_families": ["qwen2"],
                "model_type": "1.5B",
                "file_type": "Q4_K_M",
                "architecture": "amd64",
                "os": "linux",
                "rootfs": {"type": "layers", "diff_ids": []},
            }
        elif "license12345" in url:
            resp.content = b"Ollama Model License"
        elif "system12345" in url:
            resp.content = b"System prompt"
        elif "template12345" in url:
            resp.content = b"Template config"
        else:
            resp.iter_content.return_value = [b"ollamadata1", b"ollamadata2"]

        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    mock_request = MagicMock()
    mock_request.url = "http://localhost/api/pull"

    with (
        patch(
            "core.model.storage_helpers.OllamaPullSupplier.requests.get",
            side_effect=mock_get,
        ),
        patch(
            "core.model.storage_helpers.OllamaFileSystem.requests.get",
            side_effect=mock_get,
        ),
        patch(
            "core.model.storage_helpers.model_pull.requests.get", side_effect=mock_get
        ),
    ):
        res = pull_model(request=mock_request, pull_supplier=supplier)
        test_context["result"] = res
        test_context["supplier"] = supplier
        test_context["model_name"] = model_name
        test_context["tag"] = tag


@then(
    "the Ollama manifest, config, blobs, and symbolic links should be correctly saved to disk"
)
def check_ollama_saved(test_context):
    models_dir = Path(test_context.get("models_dir"))
    model_name = test_context.get("model_name")
    tag = test_context.get("tag")

    # 1. Manifest
    manifest_path = models_dir / "manifests" / model_name / tag
    assert manifest_path.exists()

    # 2. Blobs
    blobs_dir = models_dir / "blobs"
    assert blobs_dir.exists()
    assert (blobs_dir / "sha256-cfg12345").exists()
    assert (blobs_dir / "sha256-sha256:system12345").exists()
    assert (blobs_dir / "sha256-sha256:template12345").exists()
    assert (blobs_dir / "sha256-model12345").exists()

    # 3. Symbolic links
    links_dir = models_dir / "manifests" / model_name / f".{tag}"
    assert links_dir.exists()
    assert (links_dir / "model").is_symlink()
    assert (links_dir / "system").is_symlink()
    assert (links_dir / "template").is_symlink()


# --- Scenario 7 Steps ---
@then("the pull operation should return an error")
def check_pull_error(test_context):
    assert str(test_context.get("result")).startswith("Error")


@then("the model lock should be released")
def check_lock_released(test_context):
    models_dir = Path(test_context.get("models_dir"))
    model_name = test_context.get("model_name")
    lock_file = models_dir / model_name / "lock"
    assert not lock_file.exists()


@then("any partially downloaded model files should be removed from disk")
def check_partially_downloaded_removed(test_context):
    models_dir = Path(test_context.get("models_dir"))
    blobs_dir = models_dir / "blobs"
    fail_blob = blobs_dir / "sha256-fail-sha256"
    assert not fail_blob.exists()
