import json
from pytest_bdd import scenarios, given, when, then, parsers
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric

# Link to feature file
scenarios("../features/openai_chat.feature")


@given("the fastapi-rkllama application is running", target_fixture="app_state")
def app_is_running():
    # The application is already running in-process via FastAPITestClient
    return {}


@when(
    parsers.parse(
        'a chat completion request is sent with prompt "{prompt}" and model "{model}"'
    ),
    target_fixture="response_data",
)
def send_chat_completion(api_client, prompt, model, app_state):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    app_state["prompt"] = prompt
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        data = response.json()
        app_state["response_json"] = data
        app_state["output"] = data["choices"][0]["message"]["content"]
    else:
        app_state["output"] = ""
    return app_state


@when(
    parsers.parse(
        'a streaming chat completion request is sent with prompt "{prompt}" and model "{model}"'
    ),
    target_fixture="response_data",
)
def send_streaming_chat_completion(api_client, prompt, model, app_state):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    app_state["prompt"] = prompt
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if line_str.startswith("data: "):
                data_part = line_str[6:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunks.append(json.loads(data_part))
                except Exception:
                    pass
        app_state["streaming_chunks"] = chunks
    return app_state


@when(
    parsers.parse(
        'a chat completion request is sent with system prompt "{system_prompt}" and user prompt "{user_prompt}" and model "{model}"'
    ),
    target_fixture="response_data",
)
def send_chat_completion_with_system(
    api_client, system_prompt, user_prompt, model, app_state
):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    app_state["prompt"] = user_prompt
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        data = response.json()
        app_state["response_json"] = data
        app_state["output"] = data["choices"][0]["message"]["content"]
    else:
        app_state["output"] = ""
    return app_state


@when(
    parsers.parse(
        "an invalid chat completion request is sent with invalid temperature {temperature}"
    ),
    target_fixture="response_data",
)
def send_invalid_chat_completion(api_client, temperature, app_state):
    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": float(temperature),
        "stream": False,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    app_state["status_code"] = response.status_code
    return app_state


@then(parsers.parse("the API should return a status code of {status_code:d}"))
def check_status_code(response_data, status_code):
    assert response_data["status_code"] == status_code


@then("the response should contain the chat completion content")
def check_response_content(response_data):
    assert "choices" in response_data["response_json"]
    assert len(response_data["response_json"]["choices"]) > 0
    assert "message" in response_data["response_json"]["choices"][0]
    assert "content" in response_data["response_json"]["choices"][0]["message"]
    assert len(response_data["output"]) > 0


@then("the streaming chunks should be successfully parsed to build the final response")
def parse_streaming_chunks(response_data):
    chunks = response_data.get("streaming_chunks", [])
    assert len(chunks) > 0, "No streaming chunks received"

    # Reassemble the content
    content_parts = []
    for chunk in chunks:
        choices = chunk.get("choices", [])
        if choices:
            # choices is a list, delta is dict inside choice
            choice = choices[0]
            delta = choice.get("delta", {})
            if "content" in delta:
                content_parts.append(delta["content"])

    full_content = "".join(content_parts).strip()
    assert len(full_content) > 0, "Reassembled content is empty"
    response_data["output"] = full_content


@then("the response relevancy should be evaluated as successful by DeepEval")
def evaluate_relevancy(response_data, deepeval_model):
    user_input = response_data["prompt"]
    model_output = response_data["output"]

    test_case = LLMTestCase(input=user_input, actual_output=model_output)

    metric = AnswerRelevancyMetric(threshold=0.5, model=deepeval_model)
    assert_test(test_case, [metric])


@then("the streaming response relevancy should be evaluated as successful by DeepEval")
def evaluate_streaming_relevancy(response_data, deepeval_model):
    evaluate_relevancy(response_data, deepeval_model)


UnitTests = True
