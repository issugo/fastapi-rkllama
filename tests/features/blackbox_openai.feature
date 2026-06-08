Feature: OpenAI Blackbox API with Model Management
  As a client of the fastapi-rkllama server
  I want to interact with the OpenAI API for chat and completions
  So that I can load models, list them, chat, and generate completions

  Scenario: Load model, list loaded models, chat (stream/no-stream), and generate completion
    Given the fastapi-rkllama application is running with the default dummy model
    When a request is sent to list the models via OpenAI endpoint "/v1/models"
    Then the API should return a status code of 200
    
    When a request is sent to load the default model via RKLLAMA API
    Then the load model API should return a status code of 200
    
    When a request is sent to list loaded models via RKLLAMA API
    Then the default model should be in the loaded models list
    
    When a chat completion request is sent via OpenAI with prompt "Hello" not using stream
    Then the API should return a status code of 200
    And the response relevancy should be evaluated as successful by DeepEval
    
    When a chat completion request is sent via OpenAI with prompt "Hello" using stream
    Then the API should return a status code of 200
    And the streaming response relevancy should be evaluated as successful by DeepEval
    
    When a completion request is sent via OpenAI with prompt "Hello" not using stream
    Then the API should return a status code of 200
    And the completion response relevancy should be evaluated as successful by DeepEval
