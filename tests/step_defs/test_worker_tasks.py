"""
Step definitions for testing model worker tasks behavior.
"""

import os
import json
import time
import shutil
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from pathlib import Path

from core.config.config_utils import get_path
from core.model.ModelPath import ModelPath
from core.model.ModelFile import ModelFile
from core.model.ModelConfig import FullModelParameters
from core.processing.BaseDomainId import BaseDomainId
from core.processing.workers.RkllmWorker import RkllmWorker
from core.processing.tasks.Tasks import Tasks
from core.processing.tasks import (
    AbortInferenceTask,
    ClearCacheTask,
    InferenceTask,
    UnloadModelTask,
    EmbeddingTask,
    VisionEncoderTask,
    FinishedTask,
    ErrorTask,
)

# Link to feature file
scenarios("../features/worker_tasks.feature")


@pytest.fixture
def dummy_model_for_worker():
    """
    Fixture to create a dummy model for worker testing.
    """
    models_dir = get_path("models")
    model_name = "worker-test-model"
    endpoint_dir = "qwen2-7b-rk3588"
    model_file = "qwen2-7b-rk3588-w8a8.rkllm"
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

    # 2. Create dummy .metadata file
    metadata_path = f"{file_path}.metadata"
    metadata_content = {
        "name": "qwen2",
        "architecture": "qwen2",
        "quantization": "w8a8",
        "parameters": 7000000000,
        "context_length": 16384,
        "temperature": 0.7,
        "model_type": "RKLLM",
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
        "lfs": {"size": 100, "sha256": "dummy-sha256", "pointer_size": 100},
        "last_commit": None,
        "security": None,
    }
    with open(hf_file_info_path, "w") as f:
        json.dump(hf_file_info_content, f)

    # 4. Create dummy ModelInfo.json file
    hf_model_info_path = os.path.join(full_dir, "ModelInfo.json")
    hf_model_info_content = {
        "id": model_id,
        "private": False,
        "tags": ["qwen2", "rkllm"],
        "downloads": 10,
        "likes": 5,
        "modelId": model_id,
        "author": "worker-test-model",
        "sha": "dummy-sha",
        "lastModified": "2024-01-01T00:00:00.000Z",
        "gated": False,
        "disabled": False,
        "config": {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "tokenizer_config": {},
        },
        "cardData": {
            "base_model": ["qwen/qwen2"],
            "tags": ["rkllm"],
            "params": 7000000000,
        },
        "siblings": [],
        "createdAt": "2024-01-01T00:00:00.000Z",
        "usedStorage": 100,
        "languages": ["en"],
    }
    with open(hf_model_info_path, "w") as f:
        json.dump(hf_model_info_content, f)

    # 5. Create dummy .config file
    config_path = f"{file_path}.config"
    with open(config_path, "w") as f:
        f.write(f"# FROM={endpoint_dir}/{model_file}\n")
        f.write('{ "huggingface_path": "gpt2" }\n')

    yield model_id

    # Cleanup
    shutil.rmtree(full_dir, ignore_errors=True)


@pytest.fixture
def test_context():
    """
    Context dictionary to share state between steps.
    """
    return {}


@given("a model worker is running", target_fixture="worker_ctx")
def worker_is_running(dummy_model_for_worker, test_context):
    os.environ["RKLLAMA_SIMULATE"] = "true"

    model_path = ModelPath.from_model_id(dummy_model_for_worker)
    modelfile = ModelFile.load(model_path=model_path)
    
    params = FullModelParameters(
        enable_thinking=False,
        max_new_tokens=1024,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        mirostat=False,
        mirostat_tau=3.0,
        mirostat_eta=0.001
    )
    
    worker = RkllmWorker(modelfile, params, 1)
    proc = worker.create_worker_process()
    assert proc is not None
    
    test_context["worker"] = worker
    test_context["process"] = proc
    return test_context


@pytest.fixture(autouse=True)
def cleanup_worker(test_context):
    yield
    worker = test_context.get("worker")
    if worker:
        if worker.process and worker.process.is_alive():
            try:
                worker.task_q.put(UnloadModelTask())
                worker.process.join(timeout=1)
            except Exception:
                pass
            if worker.process.is_alive():
                worker.process.terminate()


@when("I send an AbortInferenceTask to the worker")
def send_abort(worker_ctx):
    worker = worker_ctx["worker"]
    worker.task_q.put(AbortInferenceTask())


@then("the worker should process the abort successfully without errors")
def check_abort_success(worker_ctx):
    worker = worker_ctx["worker"]
    # Check that the process did not crash and is still running
    assert worker.process.is_alive()
    # Check that result queue is empty
    assert worker.result_q.empty()


@when("I send a ClearCacheTask to the worker")
def send_clear_cache(worker_ctx):
    worker = worker_ctx["worker"]
    worker.task_q.put(ClearCacheTask())


@then("the worker should process the clear cache successfully without errors")
def check_clear_cache_success(worker_ctx):
    worker = worker_ctx["worker"]
    # Check that the process did not crash and is still running
    assert worker.process.is_alive()
    # Check that result queue is empty
    assert worker.result_q.empty()


@when('I send an InferenceTask with prompt "Hi" to the worker')
def send_inference(worker_ctx):
    worker = worker_ctx["worker"]
    from core.backends.rkllm.classes import RKLLMInferMode, RKLLMInputType
    # RkllmSimuBackend expects prompt_tokens. Let's pass mock token IDs.
    task = InferenceTask(
        RKLLMInferMode.RKLLM_INFER_GENERATE,
        RKLLMInputType.RKLLM_INPUT_TOKEN,
        [101, 102]
    )
    worker.task_q.put(task)


@then("the worker should return generated tokens followed by a finished status")
def check_inference_results(worker_ctx):
    worker = worker_ctx["worker"]
    tokens = []
    finished = False
    
    start_time = time.time()
    while time.time() - start_time < 15:
        if not worker.result_q.empty():
            res = worker.result_q.get()
            if res == Tasks.WORKER_TASK_FINISHED:
                finished = True
                break
            else:
                tokens.append(res)
        else:
            time.sleep(0.05)
            
    assert finished, "Inference did not return a finished status"
    assert len(tokens) > 0, "No simulated tokens were generated"


@when("I send an UnloadModelTask to the worker")
def send_unload(worker_ctx):
    worker = worker_ctx["worker"]
    worker.task_q.put(UnloadModelTask())


@then("the worker process should terminate")
def check_unload_success(worker_ctx):
    worker = worker_ctx["worker"]
    worker.process.join(timeout=5)
    assert not worker.process.is_alive(), "Worker process did not terminate after UnloadModelTask"


@when("I send an EmbeddingTask to the worker")
def send_embedding(worker_ctx):
    worker = worker_ctx["worker"]
    worker.task_q.put(EmbeddingTask())


@when("I send a VisionEncoderTask to the worker")
def send_vision_encoder(worker_ctx):
    worker = worker_ctx["worker"]
    worker.task_q.put(VisionEncoderTask())


@when("I send a FinishedTask to the worker")
def send_finished(worker_ctx):
    worker = worker_ctx["worker"]
    worker.task_q.put(FinishedTask())


@when("I send an ErrorTask to the worker")
def send_error(worker_ctx):
    worker = worker_ctx["worker"]
    worker.task_q.put(ErrorTask())


@then(parsers.parse('the worker should return an unknown task message followed by a finished status for "{task_name}"'))
def check_unknown_task_result(worker_ctx, task_name):
    worker = worker_ctx["worker"]
    msg = None
    finished = False
    
    start_time = time.time()
    while time.time() - start_time < 5:
        if not worker.result_q.empty():
            res = worker.result_q.get()
            if res == Tasks.WORKER_TASK_FINISHED:
                finished = True
                break
            else:
                msg = res
        else:
            time.sleep(0.05)
            
    assert finished, f"Task {task_name} did not return finished status"
    assert msg == f"Unknown task: {task_name}", f"Expected unknown task message for {task_name}, got {msg}"
