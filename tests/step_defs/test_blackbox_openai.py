"""
This module contains BDD step definitions for testing the OpenAI blackbox functionality.
It covers model loading via RKLLAMA API, listing models, and testing chat/completions endpoints.
"""

import os
import json
import logging
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from core.config.config_utils import get_path

logger = logging.getLogger(__name__)

# Link to feature file
scenarios("../features/blackbox_openai.feature")


@pytest.fixture
def default_dummy_model():
    """
    Fixture to create a default dummy model for blackbox testing.
    Uses the HuggingFace model specified in the rules.
    """
    logger.info("Method default_dummy_model called with parameters: None")
    models_dir = get_path("models")
    model_name = "dulimov"
    endpoint_dir = "Qwen3-4B-rk3588-1.2.1-unsloth-16k"
    model_file = "Qwen3-4B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm"
    model_id = f"{model_name}/{endpoint_dir}/{model_file}"

    full_dir = os.path.join(models_dir, model_name)
    file_dir = os.path.join(full_dir, endpoint_dir)
    os.makedirs(file_dir, exist_ok=True)

    # 1. Create dummy .rkllm file
    file_path = os.path.join(file_dir, model_file)
    with open(file_path, "wb") as f:
        f.write(b"dummy model content")

    # 1b. Create dummy Modelfile
    modelfile_path = os.path.join(full_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(f"FROM {endpoint_dir}/{model_file}\n")

    # 2. Create dummy .metadata file
    metadata_path = f"{file_path}.metadata"
    metadata_content = {
        "name": "Qwen3",
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
        "author": "dulimov",
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
        f.write(f"# FROM={endpoint_dir}/{model_file}\n")
        f.write('{ "huggingface_path": "gpt2" }\n')

    yield model_id

    # Cleanup
    try:
        os.remove(file_path)
        os.remove(modelfile_path)
        os.remove(metadata_path)
        os.remove(hf_file_info_path)
        os.remove(hf_model_info_path)
        os.remove(config_path)
        os.removedirs(file_dir)
    except OSError:
        pass


@given(
    "the fastapi-rkllama application is running with the default dummy model",
    target_fixture="app_state",
)
def app_is_running(default_dummy_model):
    """
    Initializes the BDD test state with the default dummy model.
    """
    logger.info(
        f"Method app_is_running called with parameters: default_dummy_model={default_dummy_model}"
    )
    return {"model_id": default_dummy_model}


@when(
    parsers.parse(
        'a request is sent to list the models via OpenAI endpoint "{endpoint}"'
    )
)
def list_models(api_client, app_state, endpoint):
    """
    Sends a GET request to the given OpenAI endpoint to list available models.
    """
    logger.info(
        f"Method list_models called with parameters: api_client={api_client}, app_state={app_state}, endpoint={endpoint}"
    )
    response = api_client.get(endpoint)
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()
    else:
        app_state["response_json"] = None


@then(parsers.parse("the API should return a status code of {status_code:d}"))
def check_status_code(app_state, status_code):
    """
    Asserts that the stored status code equals the expected one.
    """
    logger.info(
        f"Method check_status_code called with parameters: app_state={app_state}, status_code={status_code}"
    )
    assert app_state.get("status_code") == status_code


@when("a request is sent to load the default model via RKLLAMA API")
def load_default_model(api_client, app_state):
    """
    Sends a POST request to load the default model using RKLLAMA API.
    """
    logger.info(
        f"Method load_default_model called with parameters: api_client={api_client}, app_state={app_state}"
    )
    payload = {"model_name": app_state["model_id"]}
    response = api_client.post("/load_model", json=payload)
    app_state["load_status_code"] = response.status_code


@then(
    parsers.parse("the load model API should return a status code of {status_code:d}")
)
def check_load_status_code(app_state, status_code):
    """
    Asserts that the model loading status code equals the expected one.
    """
    logger.info(
        f"Method check_load_status_code called with parameters: app_state={app_state}, status_code={status_code}"
    )
    assert app_state.get("load_status_code") == status_code


@when("a request is sent to list loaded models via RKLLAMA API")
def list_loaded_models_rkllama(api_client, app_state):
    """
    Sends a GET request to the RKLLAMA endpoint to list loaded models.
    """
    logger.info(
        f"Method list_loaded_models_rkllama called with parameters: api_client={api_client}, app_state={app_state}"
    )
    response = api_client.get("/current_models")
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()
    else:
        app_state["response_json"] = None


@then("the default model should be in the loaded models list")
def check_model_in_loaded_list(app_state):
    """
    Checks if the default model id exists within the loaded models list.
    """
    logger.info(
        f"Method check_model_in_loaded_list called with parameters: app_state={app_state}"
    )
    response_json = app_state.get("response_json")
    assert response_json is not None
    models = response_json.get("models", []) if isinstance(response_json, dict) else []
    model_id = app_state["model_id"]
    assert any(
        m.get("model") == model_id or m.get("name") == model_id for m in models
    ), f"Model {model_id} not in loaded models: {models}"


@when(
    parsers.parse(
        'a chat completion request is sent via OpenAI with prompt "{prompt}" not using stream'
    )
)
def chat_request_no_stream(api_client, app_state, prompt):
    """
    Sends a non-streaming chat completion request.
    """
    logger.info(
        f"Method chat_request_no_stream called with parameters: api_client={api_client}, app_state={app_state}, prompt={prompt}"
    )
    payload = {
        "model": app_state["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()
        app_state["prompt"] = prompt
        app_state["output"] = app_state["response_json"]["choices"][0]["message"][
            "content"
        ]


@then("the response relevancy should be evaluated as successful by DeepEval")
def evaluate_response(app_state, deepeval_model):
    """
    Uses DeepEval to calculate the AnswerRelevancyMetric for the given response.
    """
    logger.info(
        f"Method evaluate_response called with parameters: app_state={app_state}, deepeval_model={deepeval_model}"
    )
    user_input = app_state["prompt"]
    model_output = app_state["output"]
    test_case = LLMTestCase(input=user_input, actual_output=model_output)
    metric = AnswerRelevancyMetric(threshold=0.5, model=deepeval_model)
    assert_test(test_case, [metric])


@when(
    parsers.parse(
        'a chat completion request is sent via OpenAI with prompt "{prompt}" using stream'
    )
)
def chat_request_stream(api_client, app_state, prompt):
    """
    Sends a streaming chat completion request and parses the chunks.
    """
    logger.info(
        f"Method chat_request_stream called with parameters: api_client={api_client}, app_state={app_state}, prompt={prompt}"
    )
    payload = {
        "model": app_state["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    app_state["status_code"] = response.status_code
    app_state["prompt"] = prompt

    if response.status_code == 200:
        content = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8") if isinstance(line, bytes) else line
                if decoded_line.startswith("data: ") and not decoded_line.endswith(
                    "[DONE]"
                ):
                    data_str = decoded_line[6:]
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                content += delta["content"]
                    except json.JSONDecodeError:
                        pass
        app_state["output"] = content


@then("the streaming response relevancy should be evaluated as successful by DeepEval")
def evaluate_streaming_response(app_state, deepeval_model):
    """
    Uses DeepEval to calculate the AnswerRelevancyMetric for the given streaming response.
    """
    logger.info(
        f"Method evaluate_streaming_response called with parameters: app_state={app_state}, deepeval_model={deepeval_model}"
    )
    evaluate_response(app_state, deepeval_model)


@when(
    parsers.parse(
        'a completion request is sent via OpenAI with prompt "{prompt}" not using stream'
    )
)
def completion_request_no_stream(api_client, app_state, prompt):
    """
    Sends a non-streaming text completion request.
    """
    logger.info(
        f"Method completion_request_no_stream called with parameters: api_client={api_client}, app_state={app_state}, prompt={prompt}"
    )
    payload = {"model": app_state["model_id"], "prompt": prompt, "stream": False}
    response = api_client.post("/v1/completions", json=payload)
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()
        app_state["prompt"] = prompt
        app_state["output"] = app_state["response_json"]["choices"][0]["text"]


@then("the completion response relevancy should be evaluated as successful by DeepEval")
def evaluate_completion_response(app_state, deepeval_model):
    """
    Uses DeepEval to calculate the AnswerRelevancyMetric for the given completion response.
    """
    logger.info(
        f"Method evaluate_completion_response called with parameters: app_state={app_state}, deepeval_model={deepeval_model}"
    )
    evaluate_response(app_state, deepeval_model)
