import os
import sys
import ctypes
import platform
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

# Always use the rkllm_simu backend when running on an x86 architecture
machine = platform.machine().lower()
if any(arch in machine for arch in ["x86_64", "amd64", "i386", "i686", "x86"]):
    os.environ["RKLLAMA_SIMULATE"] = "true"

# Mock sys.argv to prevent Pydantic settings from parsing pytest CLI args
sys.argv = [sys.argv[0]]

# 1. Add workspace root and app directory to pythonpath
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(workspace_root / "app"))

# 2. Intercept CDLL calls to avoid crashing on non-Rockchip platforms
original_cdll = ctypes.CDLL


def mock_cdll(name, *args, **kwargs):
    if "librkllmrt.so" in name or "librkllm" in name:
        mock_lib = MagicMock()
        # Mock functions that classes.py expects:
        mock_lib.rkllm_init = MagicMock(return_value=0)
        mock_lib.rkllm_run = MagicMock(return_value=0)
        mock_lib.rkllm_set_chat_template = MagicMock(return_value=0)
        mock_lib.rkllm_set_function_tools = MagicMock(return_value=0)
        mock_lib.rkllm_destroy = MagicMock(return_value=0)
        mock_lib.rkllm_clear_kv_cache = MagicMock(return_value=0)
        mock_lib.rkllm_abort = MagicMock(return_value=0)
        mock_lib.rkllm_load_lora = MagicMock(return_value=0)
        mock_lib.rkllm_load_prompt_cache = MagicMock(return_value=0)
        return mock_lib
    return original_cdll(name, *args, **kwargs)


ctypes.CDLL = mock_cdll

# Import app after path and ctypes are set up
from app.main import app
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel
from typing import Optional


# Helper function to generate mock Pydantic data recursively
def generate_mock_for_schema(schema_class: type[BaseModel]) -> BaseModel:
    import typing

    mock_data = {}
    for name, field in schema_class.model_fields.items():
        field_type = field.annotation
        origin = typing.get_origin(field_type)

        # Handle Union types or Optional types (e.g. float | None)
        if origin is typing.Union or (
            hasattr(typing, "UnionType") and origin is typing.UnionType
        ):
            sub_types = [t for t in field_type.__args__ if t is not type(None)]
            if sub_types:
                field_type = sub_types[0]
                origin = typing.get_origin(field_type)

        if origin is typing.Literal:
            mock_data[name] = field_type.__args__[0]
        # Handle list/set/tuple
        elif origin is list or origin is set or origin is tuple or field_type is list:
            # Check element type
            if hasattr(field_type, "__args__") and field_type.__args__:
                elem_type = field_type.__args__[0]
                # If elem_type is a Union/Optional, unwrap it
                elem_origin = typing.get_origin(elem_type)
                if elem_origin is typing.Union or (
                    hasattr(typing, "UnionType") and elem_origin is typing.UnionType
                ):
                    elem_sub_types = [
                        t for t in elem_type.__args__ if t is not type(None)
                    ]
                    if elem_sub_types:
                        elem_type = elem_sub_types[0]
                        elem_origin = typing.get_origin(elem_type)

                if isinstance(elem_type, type) and issubclass(elem_type, BaseModel):
                    mock_data[name] = [generate_mock_for_schema(elem_type)]
                elif elem_origin is typing.Literal:
                    mock_data[name] = [elem_type.__args__[0]]
                elif elem_type == int:
                    mock_data[name] = [1]
                elif elem_type == float:
                    mock_data[name] = [0.95]
                elif elem_type == bool:
                    mock_data[name] = [True]
                else:
                    mock_data[name] = ["The output is relevant and correct."]
            else:
                mock_data[name] = ["The output is relevant and correct."]
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            mock_data[name] = generate_mock_for_schema(field_type)
        elif field_type == float:
            mock_data[name] = 0.95
        elif field_type == int:
            mock_data[name] = 1
        elif field_type == bool:
            mock_data[name] = True
        elif field_type == str:
            if name.lower() in ("verdict", "verdicts"):
                mock_data[name] = "yes"
            else:
                mock_data[name] = (
                    "Evaluation completed: the actual output satisfies the requirements."
                )
        else:
            mock_data[name] = None
    return schema_class(**mock_data)


# 3. Custom DeepEval Mock LLM for offline testing
class MockEvaluationLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "Mock Evaluator"):
        self.model_name = model_name

    def load_model(self):
        return self

    def get_model_name(self):
        return self.model_name

    def generate(
        self, prompt: str, schema: Optional[type[BaseModel]] = None
    ) -> BaseModel | str:
        if schema is not None:
            return generate_mock_for_schema(schema)
        return "The actual output matches expectations perfectly."

    async def a_generate(
        self, prompt: str, schema: Optional[type[BaseModel]] = None
    ) -> BaseModel | str:
        return self.generate(prompt, schema)


# 4. Pytest fixtures
@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def deepeval_model():
    return MockEvaluationLLM()
