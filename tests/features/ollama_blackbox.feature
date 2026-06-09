Feature: Ollama Blackbox API
  As a client of the fastapi-rkllama server
  I want to interact with the Ollama API
  So that I can load models, list them, chat, and generate completions

  Scenario: Load model, list loaded models, chat (stream/no-stream), and generate completion
    Given the fastapi-rkllama application is running with the default dummy model
    When a request is sent to list loaded models via Ollama API
    Then the response should be successful
    
    When a chat completion request is sent to Ollama to load the default model
    Then the response should be successful
    
    When a request is sent to list loaded models via Ollama API
    Then the default model should be in the loaded models list
    
    When a chat completion request is sent to Ollama with prompt "Hello" not using stream
    Then the response should be successful
    And the response relevancy should be evaluated as successful by DeepEval
    
    When a chat completion request is sent to Ollama with prompt "Hello" using stream
    Then the streaming response should be successful
    And the streaming response relevancy should be evaluated as successful by DeepEval
    
    When a completion request is sent to Ollama with prompt "Hello" not using stream
    Then the response should be successful
    And the completion response relevancy should be evaluated as successful by DeepEval
