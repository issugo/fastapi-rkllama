from pytest_bdd import scenarios, given, when, then, parsers

scenarios("../features/model_management.feature")


@given("the fastapi-rkllama application is running", target_fixture="app_state")
def app_is_running():
    return {}


@when(
    parsers.parse(
        'a request is sent to list the models via OpenAI endpoint "{endpoint}"'
    ),
    target_fixture="response_data",
)
def get_openai_models(api_client, endpoint, app_state):
    response = api_client.get(endpoint)
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()
    else:
        app_state["response_json"] = None
    return app_state


@when(
    parsers.parse(
        'a request is sent to list the models via RKLLAMA endpoint "{endpoint}"'
    ),
    target_fixture="response_data",
)
def get_rkllama_models(api_client, endpoint, app_state):
    response = api_client.get(endpoint)
    app_state["status_code"] = response.status_code
    if response.status_code == 200:
        app_state["response_json"] = response.json()
    else:
        app_state["response_json"] = None
    return app_state


@then(parsers.parse("the API should return a status code of {status_code:d}"))
def check_status_code(response_data, status_code):
    assert response_data["status_code"] == status_code


@then("the response should contain a list of model objects")
def check_openai_models_list(response_data):
    response_json = response_data["response_json"]
    assert isinstance(response_json, list)
    for model in response_json:
        assert "id" in model
        assert "object" in model


@then("the response should contain a dictionary of models")
def check_rkllama_models_dict(response_data):
    response_json = response_data["response_json"]
    assert isinstance(response_json, dict)
    assert "models" in response_json
    assert isinstance(response_json["models"], list)
