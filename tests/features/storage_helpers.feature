Feature: Storage Helpers Package
  As a developer and system component of fastapi-rkllama
  I want to use storage helpers for HuggingFace and Ollama model management
  So that models can be validated, fetched, cached, and cleaned up correctly

  Scenario: Validate HuggingFace paths
    Given a HuggingFace path validator is available
    When I validate the HuggingFace path "author/model-name"
    Then the path should be valid and return "author/model-name"
    When I validate the HuggingFace path "author/model-name/file.rkllm" with author "author"
    Then the path should be valid and return "author/model-name/file.rkllm"
    When I validate the HuggingFace path "invalid_path"
    Then it should raise a ValueError
    When I validate the HuggingFace path "other-author/model-name" with author "author"
    Then it should raise a ValueError

  Scenario: Generate Ollama model and blob URLs
    Given an Ollama file system utility is available
    When I request the model path for "llama3" with api flag set to True
    Then the model path should end with "/v2/library/llama3"
    When I request the model path for "llama3" with api flag set to False
    Then the model path should end with "/library/llama3"
    When I request the blob URL for digest "sha256-12345" and model "llama3"
    Then the blob URL should end with "/v2/library/llama3/blobs/sha256-12345"
    When I request the model URL for digest "sha256-54321" and model "llama3"
    Then the model URL should end with "/v2/library/llama3/blobs/sha256-54321"
    When I request the blob URL with an empty digest
    Then it should raise a ValueError

  Scenario: Retrieve and parse HuggingFace model info
    Given a mocked HuggingFace API returning model metadata for "my-author/my-model"
    And the model description contains architecture "qwen2" and quantization "int4"
    When I load the model info for HuggingFace path "my-author/my-model"
    Then the response should contain the architecture "qwen"
    And the response should contain the quantization "int4"
    And the response should contain English language in tags
    And the response should contain the license name and URL

  Scenario: Retrieve and parse Ollama model configuration
    Given a mocked Ollama registry config endpoint for model "llama3" and digest "sha256-cfg123"
    When I load the Ollama config for digest "sha256-cfg123", model "llama3", and tag "latest"
    Then the returned configuration should contain the architecture "amd64"
    And the info dictionary should have description "llama3:latest"

  Scenario: Successfully pull an RKLLM model from HuggingFace
    Given the fastapi-rkllama storage directories are prepared
    And a mocked HuggingFace HfFileSystem and download registry
    When I pull the HuggingFace model "my-author/qwen2-7b" with file "qwen2-7b-rk3588-w8a8.rkllm"
    Then the pull operation should succeed
    And the RKLLM model file and metadata should be correctly saved to disk
    And the model should be unlocked

  Scenario: Successfully pull an Ollama model
    Given the fastapi-rkllama storage directories are prepared
    And a mocked Ollama registry API for pulling
    When I pull the Ollama model "llama3" with tag "8b"
    Then the pull operation should succeed
    And the Ollama manifest, config, blobs, and symbolic links should be correctly saved to disk
    And the model should be unlocked

  Scenario: Handle download errors and cleanup
    Given the fastapi-rkllama storage directories are prepared
    And a mocked HuggingFace registry that fails during file download
    When I pull the HuggingFace model "my-author/qwen2-7b" with file "qwen2-7b-rk3588-w8a8-fail.rkllm"
    Then the pull operation should return an error
    And the model lock should be released
    And any partially downloaded model files should be removed from disk
