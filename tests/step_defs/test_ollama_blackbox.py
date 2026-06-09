import os
import json
import shutil
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from core.config.config_utils import get_path

# Link to feature file
scenarios("../features/ollama_blackbox.feature")

@pytest.fixture
def default_dummy_model():
    models_dir = get_path("models")
    # To match ModelPath.from_model_id parsing (which takes the first part before '/' as model_name)
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
        f.write(f"# HUGGINGFACE_PATH=gpt2\n")
        
    # 2. Create dummy .metadata file
    metadata_path = f"{file_path}.metadata"
    metadata_content = {
        "name": "Qwen3",
        "architecture": "qwen2",
        "quantization": "w8a8",
        "parameters": 4000000000,
        "context_length": 16384,
        "temperature": 0.7,
        "model_type": "RKLLM"
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
        "lfs": {
            "size": 100,
            "sha256": "dummy-sha256",
            "pointer_size": 100
        },
        "last_commit": None,
        "security": None
    }
    with open(hf_file_info_path, "w") as f:
        json.dump(hf_file_info_content, f)
        
    # 4. Create dummy ModelInfo.json file (must be in model_name dir, not file_dir)
    hf_model_info_path = os.path.join(full_dir, "ModelInfo.json")
    hf_model_info_content = {
        "id": "gpt2",
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
            "tokenizer_config": {}
        },
        "cardData": {
            "base_model": ["qwen/qwen2"],
            "tags": ["rkllm"],
            "params": 4000000000
        },
        "siblings": [],
        "createdAt": "2024-01-01T00:00:00.000Z",
        "usedStorage": 100,
        "languages": ["en"]
    }
    with open(hf_model_info_path, "w") as f:
        json.dump(hf_model_info_content, f)
        
    # 5. Create dummy .config file
    config_path = f"{file_path}.config"
    with open(config_path, "w") as f:
        f.write(f"# FROM={endpoint_dir}/{model_file}\n")
        f.write(f"# HUGGINGFACE_PATH=gpt2\n")
        f.write('{"huggingface_path": "gpt2"}\n')
    
    yield model_id
    
    # Cleanup
    shutil.rmtree(full_dir, ignore_errors=True)


@given("the fastapi-rkllama application is running with the default dummy model", target_fixture="app_state")
def app_is_running(default_dummy_model):
    return {"model_id": default_dummy_model}

@when("a request is sent to list loaded models via Ollama API")
def list_loaded_models(api_client, app_state):
    response = api_client.get("/api/ps")
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()
    else:
        app_state["response_json"] = None

@then("the response should be successful")
def check_successful(app_state):
    assert app_state.get("status_code") == 200

@when("a chat completion request is sent to Ollama to load the default model")
def load_default_model(api_client, app_state):
    # A generic chat request to load the model
    payload = {
        "model": app_state["model_id"],
        "messages": [{"role": "user", "content": "Load this"}],
        "stream": False
    }
    response = api_client.post("/api/chat", json=payload)
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()

@then("the default model should be in the loaded models list")
def check_model_in_ps(app_state):
    response_json = app_state.get("response_json")
    assert response_json is not None
    models = response_json.get("models", [])
    model_id = app_state["model_id"]
    assert any(m.get("name") == model_id for m in models), f"Model {model_id} not in loaded models: {models}"

@when(parsers.parse('a chat completion request is sent to Ollama with prompt "{prompt}" not using stream'))
def chat_request_no_stream(api_client, app_state, prompt):
    payload = {
        "model": app_state["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    response = api_client.post("/api/chat", json=payload)
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()
        app_state["prompt"] = prompt
        app_state["output"] = app_state["response_json"]["message"]["content"]

@then("the response relevancy should be evaluated as successful by DeepEval")
def evaluate_response(app_state, deepeval_model):
    user_input = app_state["prompt"]
    model_output = app_state["output"]
    test_case = LLMTestCase(input=user_input, actual_output=model_output)
    metric = AnswerRelevancyMetric(threshold=0.5, model=deepeval_model)
    assert_test(test_case, [metric])

@when(parsers.parse('a chat completion request is sent to Ollama with prompt "{prompt}" using stream'))
def chat_request_stream(api_client, app_state, prompt):
    payload = {
        "model": app_state["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    response = api_client.post("/api/chat", json=payload)
    app_state["status_code"] = response.status_code
    app_state["stream_response"] = response
    app_state["prompt"] = prompt

@then("the streaming response should be successful")
def check_streaming_response(app_state):
    assert app_state.get("status_code") == 200
    response = app_state["stream_response"]
    content = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8") if isinstance(line, bytes) else line
            data = json.loads(decoded_line)
            if "message" in data and "content" in data["message"]:
                content += data["message"]["content"]
            if data.get("done", False):
                break
    assert len(content) > 0
    app_state["output"] = content

@then("the streaming response relevancy should be evaluated as successful by DeepEval")
def evaluate_streaming_response(app_state, deepeval_model):
    user_input = app_state["prompt"]
    model_output = app_state["output"]
    test_case = LLMTestCase(input=user_input, actual_output=model_output)
    metric = AnswerRelevancyMetric(threshold=0.5, model=deepeval_model)
    assert_test(test_case, [metric])

@when(parsers.parse('a completion request is sent to Ollama with prompt "{prompt}" not using stream'))
def completion_request_no_stream(api_client, app_state, prompt):
    payload = {
        "model": app_state["model_id"],
        "prompt": prompt,
        "stream": False
    }
    response = api_client.post("/api/generate", json=payload)
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()
        app_state["prompt"] = prompt
        app_state["output"] = app_state["response_json"]["response"]

@then("the completion response relevancy should be evaluated as successful by DeepEval")
def evaluate_completion_response(app_state, deepeval_model):
    user_input = app_state["prompt"]
    model_output = app_state["output"]
    test_case = LLMTestCase(input=user_input, actual_output=model_output)
    metric = AnswerRelevancyMetric(threshold=0.5, model=deepeval_model)
    assert_test(test_case, [metric])