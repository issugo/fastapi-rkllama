import datetime
import os
from enum import Enum
from os import stat_result
from typing import Any, Optional, List

from pydantic import BaseModel, Field

from core.model import logger
from core.model.ModelPath import ModelPath, str_parameters_size
from core.model.ModelType import ModelType
from core.model.converter.quantization_constants import ollama_quant_mapping
from core.model.models_constants import (
    UNKNOWN_VAL_STR,
    LANGUAGE_DEFAULT,
    MODEL_WITH_TOOLS,
)
from core.model.suppliers_model_info import (
    OllamaModelDetails,
    OllamaModelInfo,
    HFModelInfo,
)

"""
devstral:latest (from) sample for OllamaModelConfig
config_data = {
    "model_format": "gguf",
    "model_family": "llama",
    "model_families": ["llama"],
    "model_type": "23.6B",
    "file_type": "Q4_K_M",
    "architecture": "amd64",
    "os": "linux",
    "rootfs": {
        "type": "layers",
        "diff_ids": [
            "sha256:b3a2c9a8fef9be8d2ef951aecca36a36b9ea0b70abe9359eab4315bf4cd9be01",
            "sha256:6db27cd4e277c91264572b9c899c1980daa8dea11e902f0070a6f4763f3d13c8",
            "sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
            "sha256:5725afc40acd80cbeefba61e41cf50eb7924f6ed2fe6aec2dc6fa0e9f2c396d1"
        ]
    }
}

"""

"""
class OllamaModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    modified_at: str = Field(..., description="Last modification time")
    size: int = Field(..., description="Model size in bytes")
    digest: str = Field(..., description="Model digest")
    details: Dict[str, Any] = Field(..., description="Model details")
"""

"""
hf_modelinfo_sample: dict = {'_id': '6832397972750614b899eba5',
                             'id': 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k',
                             'private': False,
                             'tags': ['qwen3', 'unsloth', 'base_model:Qwen/Qwen3-1.7B',
                                      'base_model:finetune:Qwen/Qwen3-1.7B', 'region:us', 'rockchip', 'rk3588'],
                             'downloads': 3,
                             'likes': 1,
                             'modelId': 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k',
                             'author': 'dulimov',
                             'sha': 'd2dfbce3a30f60b584734d4e06cf145ace54a578',
                             'lastModified': '2025-05-24T21:37:33.000Z',
                             'gated': False,
                             'disabled': False,
                             'model-index': None,
                             'config': {'architectures': ['Qwen3ForCausalLM'], 'model_type': 'qwen3',
                                        'tokenizer_config': {'bos_token': None, 'eos_token': '<|im_end|>',
                                                             'pad_token': '<|vision_pad|>', 'unk_token': None},
                                        'chat_template_jinja': '{%- if tools %}\n {{- \'<|im_start|>system\\n\' }}\n {%- if messages[0].role == \'system\' %}\n {{- messages[0].content + \'\\n\\n\' }}\n {%- endif %}\n {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n {%- for tool in tools %}\n {{- "\\n" }}\n {{- tool | tojson }}\n {%- endfor %}\n {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n {%- if messages[0].role == \'system\' %}\n {{- \'<|im_start|>system\\n\' + messages[0].content + \'<|im_end|>\\n\' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for forward_message in messages %}\n {%- set index = (messages|length - 1) - loop.index0 %}\n {%- set message = messages[index] %}\n {%- set current_content = message.content if message.content is not none else \'\' %}\n {%- set tool_start = \'<tool_response>\' %}\n {%- set tool_start_length = tool_start|length %}\n {%- set start_of_message = current_content[:tool_start_length] %}\n {%- set tool_end = \'</tool_response>\' %}\n {%- set tool_end_length = tool_end|length %}\n {%- set start_pos = (current_content|length) - tool_end_length %}\n {%- if start_pos < 0 %}\n {%- set start_pos = 0 %}\n {%- endif %}\n {%- set end_of_message = current_content[start_pos:] %}\n {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}\n {%- set ns.multi_step_tool = false %}\n {%- set ns.last_query_index = index %}\n {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}\n {%- elif message.role == "assistant" %}\n {%- set content = message.content %}\n {%- set reasoning_content = \'\' %}\n {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n {%- set reasoning_content = message.reasoning_content %}\n {%- else %}\n {%- if \'</think>\' in message.content %}\n {%- set content = (message.content.split(\'</think>\')|last).lstrip(\'\\n\') %}\n {%- set reasoning_content = (message.content.split(\'</think>\')|first).rstrip(\'\\n\') %}\n {%- set reasoning_content = (reasoning_content.split(\'<think>\')|last).lstrip(\'\\n\') %}\n {%- endif %}\n {%- endif %}\n {%- if loop.index0 > ns.last_query_index %}\n {%- if loop.last or (not loop.last and reasoning_content) %}\n {{- \'<|im_start|>\' + message.role + \'\\n<think>\\n\' + reasoning_content.strip(\'\\n\') + \'\\n</think>\\n\\n\' + content.lstrip(\'\\n\') }}\n {%- else %}\n {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n {%- endif %}\n {%- else %}\n {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n {%- endif %}\n {%- if message.tool_calls %}\n {%- for tool_call in message.tool_calls %}\n {%- if (loop.first and content) or (not loop.first) %}\n {{- \'\\n\' }}\n {%- endif %}\n {%- if tool_call.function %}\n {%- set tool_call = tool_call.function %}\n {%- endif %}\n {{- \'<tool_call>\\n{"name": "\' }}\n {{- tool_call.name }}\n {{- \'", "arguments": \' }}\n {%- if tool_call.arguments is string %}\n {{- tool_call.arguments }}\n {%- else %}\n {{- tool_call.arguments | tojson }}\n {%- endif %}\n {{- \'}\\n</tool_call>\' }}\n {%- endfor %}\n {%- endif %}\n {{- \'<|im_end|>\\n\' }}\n {%- elif message.role == "tool" %}\n {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\n {{- \'<|im_start|>user\' }}\n {%- endif %}\n {{- \'\\n<tool_response>\\n\' }}\n {{- message.content }}\n {{- \'\\n</tool_response>\' }}\n {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n {{- \'<|im_end|>\\n\' }}\n {%- endif %}\n {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n {{- \'<|im_start|>assistant\\n\' }}\n {%- if enable_thinking is defined and enable_thinking is false %}\n {{- \'<think>\\n\\n</think>\\n\\n\' }}\n {%- endif %}\n{%- endif %}'},
                             'cardData': {'base_model': ['Qwen/Qwen3-1.7B'], 'tags': ['unsloth'], 'params': 1700000000},
                             'siblings': [{'rfilename': '.gitattributes'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'README.md'}, {'rfilename': 'added_tokens.json'},
                                          {'rfilename': 'chat_template.jinja'}, {'rfilename': 'config.json'},
                                          {'rfilename': 'generation_config.json'}, {'rfilename': 'merges.txt'},
                                          {'rfilename': 'special_tokens_map.json'}, {'rfilename': 'tokenizer.json'},
                                          {'rfilename': 'tokenizer_config.json'}, {'rfilename': 'vocab.json'}],
                             'spaces': [],
                             'createdAt': '2025-05-24T21:26:17.000Z',
                             'usedStorage': 17308806896,
                             'languages': ['en']}

test:HFModelInfo = HFModelInfo(**hf_modelinfo_sample)
print(test)
"""


class ModelDetails(OllamaModelDetails):
    parameter_size: str  # ex: 3B
    quantization_level: str

    _model_families: List[str] = None  # ex: ['llama']

    @property
    def model_families(self) -> List[str]:
        if self._model_families is None:
            self._model_families = [self.model_family]
        return self._model_families

    @staticmethod
    def from_model_path(model_path: ModelPath) -> Any:
        """
        Extract model parameter size and quantization type from model name

        Args:
            model_name: Model name or file path

        Returns:
            Dictionary with parameter_size and quantization_level
        """
        # Initialize default values
        # TODO: set model_format and model_family, then remove Optionals from OllamaModelDetails
        details = ModelDetails(
            model_format=UNKNOWN_VAL_STR,
            model_family=UNKNOWN_VAL_STR,
            parameter_size=UNKNOWN_VAL_STR,
            quantization_level=UNKNOWN_VAL_STR,
        )

        # Remove path and extension if present
        if isinstance(model_path.model_name, str):
            basename = os.path.basename(model_path.model_name).replace(
                model_path.model_type.get_extension(), ""
            )
        else:
            basename = str(model_name)

        model_format = model_path.get_model_format()
        if model_format:
            details.model_format = model_format

        model_family = model_path.get_model_family()
        if model_family:
            details.model_family = model_family

        parameter_size = ModelPath.get_parameter_size(basename)
        if parameter_size:
            details.parameter_size = parameter_size

        ollama_quant_level = ModelPath.get_ollama_quant_level(basename)
        if ollama_quant_level:
            details.quantization_level = ollama_quant_level

        return details

    def gen_endpoint_model_file_name(
        self,
        model_name: str,
        model_type: ModelType,
    ) -> str:
        endpoint_model_file = model_name
        parameter_size = ModelPath.get_parameter_size(model_name)
        if parameter_size and parameter_size != UNKNOWN_VAL_STR:
            if parameter_size != self.parameter_size:
                endpoint_model_file = f"{endpoint_model_file}_{self.parameter_size}"
        if self.quantization_level and self.quantization_level != UNKNOWN_VAL_STR:
            endpoint_model_file = f"{endpoint_model_file}_{ollama_quant_mapping.get(self.quantization_level)}"
        return f"{endpoint_model_file}.{model_type.get_extension()}"


class ModelInfoTag(str, Enum):
    chat = "chat"
    text_generation = "text-generation"


class DummyStatResult:
    st_size: int
    st_atime: float
    st_ctime: float
    st_mtime: float

    def __init__(self, st_size: int, st_atime: float, st_ctime: float, st_mtime: float):
        self.st_size = st_size
        self.st_atime = st_atime
        self.st_ctime = st_ctime
        self.st_mtime = st_mtime


class ModelInfo(BaseModel):
    """
    ModelInfo contains only model file stats,
    and nothing in relation with model content configuration

    model configuration (context_length, max_tokens, etc.) is in ModelMetadata
    """

    name: str  # Use simplified name like qwen:3b
    model: str  # Match Ollama's format
    created_at_dt: datetime.datetime
    modified_at_dt: datetime.datetime
    author: Optional[str] = Field(default=None, description="ex: dulimov")
    size: int
    digest: str
    details: ModelDetails
    model_type: ModelType
    # tag default is ["chat", "text-generation"]
    tags: Optional[List[str]] = Field(
        default=None,
        description="ex: ['qwen3', 'unsloth', 'base_model:Qwen/Qwen3-1.7B', 'base_model:finetune:Qwen/Qwen3-1.7B', 'region:us', 'rockchip', 'rk3588']",
    )
    languages: List[str] = Field(default=LANGUAGE_DEFAULT)
    base_model: Optional[str] = Field(default=None, description="ex: Qwen/Qwen3-1.7B")

    _ollama_model_info: OllamaModelInfo = None
    _hf_model_info: HFModelInfo = None

    _model_path: ModelPath = None

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, model_path: ModelPath):
        self._model_path = model_path

    @property
    def capabilities(self) -> List[str]:
        capabilities = ["completion"]
        if self.details.model_family in MODEL_WITH_TOOLS:
            capabilities.append("tools")
        return capabilities

    @classmethod
    def from_ollama_model_info(
        cls,
        ollama_model_info: OllamaModelInfo,
        model_path: ModelPath,
        size: int,
        digest: str,
        model_stat: stat_result | DummyStatResult,
    ):
        """
        model_format: str = Field(description="ex: gguf")
        model_family: str = Field(description="ex: llama")
        model_families: List[str]
        model_type: str  # parameter size
        file_type: str  # quantization level
        architecture: str  # ex: amd64, from processor

        sample:
        config_data = {
            "model_format": "gguf",
            "model_family": "llama",
            "model_families": ["llama"],
            "model_type": "23.6B",
            "file_type": "Q4_K_M",
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {
                "type": "layers",
                "diff_ids": [
                    "sha256:b3a2c9a8fef9be8d2ef951aecca36a36b9ea0b70abe9359eab4315bf4cd9be01",
                    "sha256:6db27cd4e277c91264572b9c899c1980daa8dea11e902f0070a6f4763f3d13c8",
                    "sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
                    "sha256:5725afc40acd80cbeefba61e41cf50eb7924f6ed2fe6aec2dc6fa0e9f2c396d1"
                ]
            }
        }

        ModelInfo:
            name: str  # Use simplified name like qwen:3b
            model: str  # Match Ollama's format
            created_at_dt: datetime.datetime
            modified_at_dt: datetime.datetime
            size: int
            digest: str
            details: ModelDetails
            model_type: ModelType

        """
        model_info = cls(
            name=model_path.model_name,
            model=ollama_model_info.model_family,
            created_at_dt=datetime.datetime.fromtimestamp(model_stat.st_ctime),
            modified_at_dt=datetime.datetime.fromtimestamp(model_stat.st_mtime),
            size=size,
            digest=digest,
            details=ModelDetails(
                model_format=ollama_model_info.model_format,
                model_family=ollama_model_info.model_family,
                parameter_size=ollama_model_info.model_type,
                quantization_level=ollama_model_info.file_type,
            ),
            model_type=model_path.model_type,
        )
        model_info.model_path = model_path
        model_info._ollama_model_info = ollama_model_info
        return model_info

    @classmethod
    def from_hf_model_info(
        cls,
        hf_model_info: HFModelInfo,
        model_path: ModelPath,
        size: int,
        digest: str,
        model_stat: stat_result | DummyStatResult,
    ):
        """
        hf_modelinfo_sample: dict = {'_id': '6832397972750614b899eba5',
                                     'id': 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k',
                                     'private': False,
                                     'tags': ['qwen3', 'unsloth', 'base_model:Qwen/Qwen3-1.7B',
                                              'base_model:finetune:Qwen/Qwen3-1.7B', 'region:us', 'rockchip', 'rk3588'],
                                     'downloads': 3,
                                     'likes': 1,
                                     'modelId': 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k',
                                     'author': 'dulimov',
                                     'sha': 'd2dfbce3a30f60b584734d4e06cf145ace54a578',
                                     'lastModified': '2025-05-24T21:37:33.000Z',
                                     'gated': False,
                                     'disabled': False,
                                     'model-index': None,
                                     'config': {'architectures': ['Qwen3ForCausalLM'], 'model_type': 'qwen3',
                                                'tokenizer_config': {'bos_token': None, 'eos_token': '<|im_end|>',
                                                                     'pad_token': '<|vision_pad|>', 'unk_token': None},
                                                'chat_template_jinja': '{%- if tools %}\n {{- \'<|im_start|>system\\n\' }}\n {%- if messages[0].role == \'system\' %}\n {{- messages[0].content + \'\\n\\n\' }}\n {%- endif %}\n {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n {%- for tool in tools %}\n {{- "\\n" }}\n {{- tool | tojson }}\n {%- endfor %}\n {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n {%- if messages[0].role == \'system\' %}\n {{- \'<|im_start|>system\\n\' + messages[0].content + \'<|im_end|>\\n\' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for forward_message in messages %}\n {%- set index = (messages|length - 1) - loop.index0 %}\n {%- set message = messages[index] %}\n {%- set current_content = message.content if message.content is not none else \'\' %}\n {%- set tool_start = \'<tool_response>\' %}\n {%- set tool_start_length = tool_start|length %}\n {%- set start_of_message = current_content[:tool_start_length] %}\n {%- set tool_end = \'</tool_response>\' %}\n {%- set tool_end_length = tool_end|length %}\n {%- set start_pos = (current_content|length) - tool_end_length %}\n {%- if start_pos < 0 %}\n {%- set start_pos = 0 %}\n {%- endif %}\n {%- set end_of_message = current_content[start_pos:] %}\n {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}\n {%- set ns.multi_step_tool = false %}\n {%- set ns.last_query_index = index %}\n {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}\n {%- elif message.role == "assistant" %}\n {%- set content = message.content %}\n {%- set reasoning_content = \'\' %}\n {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n {%- set reasoning_content = message.reasoning_content %}\n {%- else %}\n {%- if \'</think>\' in message.content %}\n {%- set content = (message.content.split(\'</think>\')|last).lstrip(\'\\n\') %}\n {%- set reasoning_content = (message.content.split(\'</think>\')|first).rstrip(\'\\n\') %}\n {%- set reasoning_content = (reasoning_content.split(\'<think>\')|last).lstrip(\'\\n\') %}\n {%- endif %}\n {%- endif %}\n {%- if loop.index0 > ns.last_query_index %}\n {%- if loop.last or (not loop.last and reasoning_content) %}\n {{- \'<|im_start|>\' + message.role + \'\\n<think>\\n\' + reasoning_content.strip(\'\\n\') + \'\\n</think>\\n\\n\' + content.lstrip(\'\\n\') }}\n {%- else %}\n {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n {%- endif %}\n {%- else %}\n {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n {%- endif %}\n {%- if message.tool_calls %}\n {%- for tool_call in message.tool_calls %}\n {%- if (loop.first and content) or (not loop.first) %}\n {{- \'\\n\' }}\n {%- endif %}\n {%- if tool_call.function %}\n {%- set tool_call = tool_call.function %}\n {%- endif %}\n {{- \'<tool_call>\\n{"name": "\' }}\n {{- tool_call.name }}\n {{- \'", "arguments": \' }}\n {%- if tool_call.arguments is string %}\n {{- tool_call.arguments }}\n {%- else %}\n {{- tool_call.arguments | tojson }}\n {%- endif %}\n {{- \'}\\n</tool_call>\' }}\n {%- endfor %}\n {%- endif %}\n {{- \'<|im_end|>\\n\' }}\n {%- elif message.role == "tool" %}\n {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\n {{- \'<|im_start|>user\' }}\n {%- endif %}\n {{- \'\\n<tool_response>\\n\' }}\n {{- message.content }}\n {{- \'\\n</tool_response>\' }}\n {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n {{- \'<|im_end|>\\n\' }}\n {%- endif %}\n {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n {{- \'<|im_start|>assistant\\n\' }}\n {%- if enable_thinking is defined and enable_thinking is false %}\n {{- \'<think>\\n\\n</think>\\n\\n\' }}\n {%- endif %}\n{%- endif %}'},
                                     'cardData': {'base_model': ['Qwen/Qwen3-1.7B'], 'tags': ['unsloth'], 'params': 1700000000},
                                     'siblings': [{'rfilename': '.gitattributes'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-1.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.5.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-0.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-0.5.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-1.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.5.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-1.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-0.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-0.5.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-1.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.5.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-1.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.5.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-1.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.0.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.5.rkllm'},
                                                  {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-1.0.rkllm'},
                                                  {'rfilename': 'README.md'}, {'rfilename': 'added_tokens.json'},
                                                  {'rfilename': 'chat_template.jinja'}, {'rfilename': 'config.json'},
                                                  {'rfilename': 'generation_config.json'}, {'rfilename': 'merges.txt'},
                                                  {'rfilename': 'special_tokens_map.json'}, {'rfilename': 'tokenizer.json'},
                                                  {'rfilename': 'tokenizer_config.json'}, {'rfilename': 'vocab.json'}],
                                     'spaces': [],
                                     'createdAt': '2025-05-24T21:26:17.000Z',
                                     'usedStorage': 17308806896,
                                     'languages': ['en']}

        test:HFModelInfo = HFModelInfo(**hf_modelinfo_sample)
        print(test)

        ModelInfo:
            name: str  # Use simplified name like qwen:3b
            model: str  # Match Ollama's format
            created_at_dt: datetime.datetime
            modified_at_dt: datetime.datetime
            size: int
            digest: str
            details: ModelDetails
            model_type: ModelType

        """
        model_type = model_path.model_type
        size_value, size_unit, _ = str_parameters_size(hf_model_info.cardData.params)
        model_info = cls(
            name=model_path.model_name,
            model=hf_model_info.config.model_type,
            created_at_dt=datetime.datetime.fromisoformat(hf_model_info.createdAt),
            modified_at_dt=datetime.datetime.fromisoformat(hf_model_info.lastModified),
            author=hf_model_info.author,
            size=size,
            digest=digest,
            details=ModelDetails(
                model_format=model_type.value.lower(),
                model_family=hf_model_info.config.model_type,
                parameter_size=f"{size_value:.2f}{size_unit.upper()}",
                quantization_level=UNKNOWN_VAL_STR,  # depends of the model, use metadata to retrieve it
            ),
            model_type=model_type,
            tags=hf_model_info.tags,
            languages=hf_model_info.languages or LANGUAGE_DEFAULT,
            base_model=(
                hf_model_info.cardData.base_model[-1]
                if hf_model_info.cardData.base_model
                else None
            ),
        )
        model_info.model_path = model_path
        model_info._hf_model_info = hf_model_info
        return model_info

    @property
    def ollama_model_info(self):
        if self._ollama_model_info is None:
            if self._model_path is not None:
                if not self._model_path.ollama_model_info_exists:
                    logger.warning(
                        f"Ollama model info not found for model {self.model}"
                    )
                    return None
            self._ollama_model_info = OllamaModelInfo.load(self.model)
        return self._ollama_model_info

    @property
    def hf_model_info(self):
        if self._hf_model_info is None:
            if self._model_path is not None:
                if not self._model_path.huggingface_model_info_exists:
                    logger.warning(
                        f"HuggingFace model info not found for model {self.model}"
                    )
                    return None
            self._hf_model_info = HFModelInfo.load(self.model)
        return self._hf_model_info
