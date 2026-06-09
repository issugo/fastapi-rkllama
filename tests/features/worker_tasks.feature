Feature: Worker Tasks Behavior
  As a developer of the fastapi-rkllama application
  I want to verify that the worker process handles all tasks correctly
  So that model loading, unloading, inference, caching, and errors behave as expected

  Scenario: Worker handles AbortInferenceTask
    Given a model worker is running
    When I send an AbortInferenceTask to the worker
    Then the worker should process the abort successfully without errors

  Scenario: Worker handles ClearCacheTask
    Given a model worker is running
    When I send a ClearCacheTask to the worker
    Then the worker should process the clear cache successfully without errors

  Scenario: Worker handles InferenceTask
    Given a model worker is running
    When I send an InferenceTask with prompt "Hi" to the worker
    Then the worker should return generated tokens followed by a finished status

  Scenario: Worker handles UnloadModelTask
    Given a model worker is running
    When I send an UnloadModelTask to the worker
    Then the worker process should terminate

  Scenario: Worker handles EmbeddingTask
    Given a model worker is running
    When I send an EmbeddingTask to the worker
    Then the worker should return an unknown task message followed by a finished status for "Tasks.WORKER_TASK_EMBEDDING"

  Scenario: Worker handles VisionEncoderTask
    Given a model worker is running
    When I send a VisionEncoderTask to the worker
    Then the worker should return an unknown task message followed by a finished status for "Tasks.WORKER_TASK_VISION_ENCODER"

  Scenario: Worker handles FinishedTask
    Given a model worker is running
    When I send a FinishedTask to the worker
    Then the worker should return an unknown task message followed by a finished status for "Tasks.WORKER_TASK_FINISHED"

  Scenario: Worker handles ErrorTask
    Given a model worker is running
    When I send an ErrorTask to the worker
    Then the worker should return an unknown task message followed by a finished status for "Tasks.WORKER_TASK_ERROR"
