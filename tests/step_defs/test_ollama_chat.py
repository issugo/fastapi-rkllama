"""
Step definitions for Ollama chat feature.

This module contains the BDD step definitions for testing the Ollama-compatible chat API.
It includes mocks for various dependencies and tests for both streaming and non-streaming
chat completions, as well as error handling.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pytest_bdd import scenarios, given, when, then, parsers
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric

from core.model.ModelConfig import FullModelParameters

# Link to feature file
scenarios("../features/ollama_chat.feature")


@pytest.fixture
def base_full_params():
    """
    Provides a base set of model parameters for testing.

    Returns:
        FullModelParameters: A set of default model parameters.
    """
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


@pytest.fixture(autouse=True)
def mock_ollama_dependencies(base_full_params):
    """
    Mocks various dependencies of the Ollama API for testing.

    This includes mocking:
    - ModelPath.from_model_id
    - ModelFile.load
    - get_worker_manager
    - ChatEndpointHandler.handle_request

    Args:
        base_full_params (FullModelParameters): The base model parameters.
    """
    from core.model.ModelFile import ModelFile
    from core.model.ModelPath import ModelPath
    from core.model.Model import Model
    from core.model.ModelType import ModelType
    from core.processing.WorkerManager import WorkerManager
    from core.api.parameters.ollama_responses import OllamaChatResponse
    from core.api.parameters.commons import Message, Role

    mock_model_path = MagicMock(spec=ModelPath)

    mock_file = MagicMock(spec=ModelFile)
    mock_file.model_id = "mock-model"
    mock_file.model_name = "mock-model"
    mock_file.SYSTEM = "You are a helpful assistant."

    mock_model = MagicMock(spec=Model)
    mock_model.model_type = ModelType.RKLLM
    mock_file.model = mock_model
    mock_file.full_model_parameters = base_full_params

    mock_wm = MagicMock(spec=WorkerManager)
    mock_worker = MagicMock()
    mock_process = MagicMock()
    mock_wm.add_worker.return_value = (mock_worker, mock_process)

    def mock_handle_request(
        model_worker,
        api_handler,
        modelfile,
        messages,
        system,
        stream,
        options,
        enable_thinking=False,
        tools=None,
        images=None,
        format_spec=None,
    ):
        """
        Mock implementation of the handle_request method.
        """
        if stream:

            async def mock_stream():
                # Yield one or more chunks in application/x-ndjson format
                yield (
                    OllamaChatResponse(
                        model=modelfile.model_name,
                        created_at="2026-06-05T08:18:14Z",
                        message=Message(
                            role=Role.ASSISTANT,
                            content="This is a mocked streaming response from the Ollama API implementation.",
                        ),
                        done=True,
                    )
                    .model_dump_json()
                    .encode()
                    + b"\n"
                )

            from starlette.responses import StreamingResponse

            return StreamingResponse(mock_stream(), media_type="application/x-ndjson")
        else:
            return OllamaChatResponse(
                model=modelfile.model_name,
                created_at="2026-06-05T08:18:14Z",
                message=Message(
                    role=Role.ASSISTANT,
                    content="This is a mocked response from the Ollama API implementation.",
                ),
                done=True,
            )

    with (
        patch(
            "api.routes.ollama.ModelPath.from_model_id",
            return_value=mock_model_path,
        ),
        patch("api.routes.ollama.ModelFile.load", return_value=mock_file),
        patch("api.routes.ollama.get_worker_manager", return_value=mock_wm),
        patch(
            "api.routes.ollama.ChatEndpointHandler.handle_request",
            side_effect=mock_handle_request,
        ),
    ):
        yield


@given("the fastapi-rkllama application is running", target_fixture="app_state")
def app_is_running():
    """
    Sets up the initial application state.
    """
    # The application is already running in-process via FastAPITestClient
    return {}


@when(
    parsers.parse(
        'a chat completion request is sent to Ollama with prompt "{prompt}" and model "{model}"'
    ),
    target_fixture="response_data",
)
def send_ollama_chat_completion(api_client, prompt, model, app_state):
    """
    Sends a non-streaming chat completion request to the Ollama API.

    Args:
        api_client (TestClient): The test client to use.
        prompt (str): The prompt to send.
        model (str): The model name.
        app_state (dict): The current application state.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = api_client.post("/api/chat", json=payload)
    app_state["prompt"] = prompt
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        data = response.json()
        app_state["response_json"] = data
        app_state["output"] = data["message"]["content"]
    else:
        app_state["output"] = ""
    return app_state


@then(parsers.parse("the Ollama API should return a status code of {status_code:d}"))
def check_ollama_status_code(response_data, status_code):
    """
    Checks the status code returned by the Ollama API.

    Args:
        response_data (dict): The data from the response.
        status_code (int): The expected status code.
    """
    assert response_data["status_code"] == status_code


@then("the response should contain the Ollama chat content")
def check_ollama_response_content(response_data):
    """
    Checks that the response contains the expected chat content.

    Args:
        response_data (dict): The data from the response.
    """
    assert "message" in response_data["response_json"]
    assert "content" in response_data["response_json"]["message"]
    assert response_data["response_json"]["message"]["role"] == "assistant"
    assert len(response_data["output"]) > 0


@then("the Ollama response relevancy should be evaluated as successful by DeepEval")
def evaluate_ollama_relevancy(response_data, deepeval_model):
    """
    Evaluates the relevancy of the response using DeepEval.

    Args:
        response_data (dict): The data from the response.
        deepeval_model (MockEvaluationLLM): The DeepEval model to use.
    """
    user_input = response_data["prompt"]
    model_output = response_data["output"]

    test_case = LLMTestCase(input=user_input, actual_output=model_output)

    metric = AnswerRelevancyMetric(threshold=0.5, model=deepeval_model)
    assert_test(test_case, [metric])


@when(
    parsers.parse(
        'a streaming chat completion request is sent to Ollama with prompt "{prompt}" and model "{model}"'
    ),
    target_fixture="response_data",
)
def send_ollama_streaming_chat_completion(api_client, prompt, model, app_state):
    """
    Sends a streaming chat completion request to the Ollama API.

    Args:
        api_client (TestClient): The test client to use.
        prompt (str): The prompt to send.
        model (str): The model name.
        app_state (dict): The current application state.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    response = api_client.post("/api/chat", json=payload)
    app_state["prompt"] = prompt
    app_state["status_code"] = response.status_code
    app_state["response"] = response
    return app_state


@then(
    "the Ollama streaming chunks should be successfully parsed to build the final response"
)
def check_ollama_streaming_chunks(response_data):
    """
    Parses the streaming chunks and builds the final response.

    Args:
        response_data (dict): The data from the response.
    """
    response = response_data["response"]
    content = ""
    chunks_count = 0

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8") if isinstance(line, bytes) else line
            data = json.loads(decoded_line)
            assert "message" in data
            assert "content" in data["message"]
            content += data["message"]["content"]
            chunks_count += 1
            if data.get("done", False):
                break

    assert chunks_count > 0
    assert len(content) > 0
    response_data["output"] = content


@then(
    "the Ollama streaming response relevancy should be evaluated as successful by DeepEval"
)
def evaluate_ollama_streaming_relevancy(response_data, deepeval_model):
    """
    Evaluates the relevancy of the streaming response using DeepEval.

    Args:
        response_data (dict): The data from the response.
        deepeval_model (MockEvaluationLLM): The DeepEval model to use.
    """
    user_input = response_data["prompt"]
    model_output = response_data["output"]

    test_case = LLMTestCase(input=user_input, actual_output=model_output)

    metric = AnswerRelevancyMetric(threshold=0.5, model=deepeval_model)
    assert_test(test_case, [metric])


@when(
    parsers.parse(
        'a chat completion request is sent to Ollama with system prompt "{system_prompt}" and user prompt "{user_prompt}" and model "{model}"'
    ),
    target_fixture="response_data",
)
def send_ollama_chat_with_system_and_user(
    api_client, system_prompt, user_prompt, model, app_state
):
    """
    Sends a chat completion request with system and user prompts to the Ollama API.

    Args:
        api_client (TestClient): The test client to use.
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        model (str): The model name.
        app_state (dict): The current application state.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    response = api_client.post("/api/chat", json=payload)
    app_state["prompt"] = user_prompt
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        data = response.json()
        app_state["response_json"] = data
        app_state["output"] = data["message"]["content"]
    else:
        app_state["output"] = ""
    return app_state


@when(
    parsers.parse(
        "an invalid chat completion request is sent to Ollama with invalid temperature {temperature}"
    ),
    target_fixture="response_data",
)
def send_ollama_invalid_chat(api_client, temperature, app_state):
    """
    Sends an invalid chat completion request with an invalid temperature.

    Args:
        api_client (TestClient): The test client to use.
        temperature (float): The invalid temperature value.
        app_state (dict): The current application state.
    """
    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    response = api_client.post("/api/chat", json=payload)
    app_state["status_code"] = response.status_code
    return app_state


# Modification Summary:
# - Added module-level docstring.
# - Added docstrings to all functions and fixtures for compliance with documentation guidelines.
# - Ensured all code modifications are documented directly in the file.
