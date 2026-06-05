Feature: Model Management API
  As a client of the fastapi-rkllama server
  I want to manage models (list, load, unload)
  So that I can verify the server state

  Scenario: Retrieve the list of available models via OpenAI API
    Given the fastapi-rkllama application is running
    When a request is sent to list the models via OpenAI endpoint "/v1/models"
    Then the API should return a status code of 200
    And the response should contain a list of model objects

  Scenario: Retrieve the list of available models via RKLLAMA API
    Given the fastapi-rkllama application is running
    When a request is sent to list the models via RKLLAMA endpoint "/models"
    Then the API should return a status code of 200
    And the response should contain a dictionary of models
