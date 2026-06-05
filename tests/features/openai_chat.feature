Feature: OpenAI Chat Completions API
  As a client of the fastapi-rkllama server
  I want to send chat prompts to the OpenAI completions endpoint
  So that I can receive responses and verify their relevancy using DeepEval

  Scenario: Request a chat completion and evaluate relevancy
    Given the fastapi-rkllama application is running
    When a chat completion request is sent with prompt "Explain the concept of gravity" and model "mock-model"
    Then the API should return a status code of 200
    And the response should contain the chat completion content
    And the response relevancy should be evaluated as successful by DeepEval

  Scenario: Request a streaming chat completion and evaluate relevancy
    Given the fastapi-rkllama application is running
    When a streaming chat completion request is sent with prompt "How does photosynthesis work?" and model "mock-model"
    Then the API should return a status code of 200
    And the streaming chunks should be successfully parsed to build the final response
    And the streaming response relevancy should be evaluated as successful by DeepEval

  Scenario: Request a chat completion with system and user messages
    Given the fastapi-rkllama application is running
    When a chat completion request is sent with system prompt "You are a helpful assistant" and user prompt "What is the capital of France?" and model "mock-model"
    Then the API should return a status code of 200
    And the response should contain the chat completion content
    And the response relevancy should be evaluated as successful by DeepEval

  Scenario: Request a chat completion with invalid parameters and receive an error
    Given the fastapi-rkllama application is running
    When an invalid chat completion request is sent with invalid temperature 3.5
    Then the API should return a status code of 422

