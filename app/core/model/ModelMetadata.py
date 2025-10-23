import json
import os
import re
from enum import Enum

from pydantic_core import from_json
from typing import Optional, List, Tuple, Any

from pydantic import BaseModel, Field

from core.backends.backend import BACKEND_SUPPORTED_LIB_VERSION
from core.config.PlatformConfig import PlatformProcessor
from core.model.ModelInfo import ModelDetails, HFModelInfo, OllamaModelInfo
from core.model.ModelPath import ModelPath, int_parameters_size
from core.model.ModelType import ModelType
from core.model import logger
from core.model.converter.quantization_constants import RK_QUANT_FORMAT, ollama_quant_mapping, OLLAMA_QUANT_FORMAT
from core.model.models_constants import MODEL_SPECS, RK_TAGS_LIST, default_context_length


class ModelMetadataFormat(str, Enum):
    SIMPLE = "simple"
    BASIC = "basic"
    COMPLETE = "complete"

    @staticmethod
    def get_format(metadata_path: str):
        if not os.path.isfile(metadata_path):
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            if "conversion_date" in metadata:
                return ModelMetadataFormat.COMPLETE
            elif "model_id" in metadata:
                return ModelMetadataFormat.BASIC
            else:
                return ModelMetadataFormat.SIMPLE

METADATA_FILENAME="metadata.json"

class ModelMetadataParameters(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stop_sequences: List[str] = ["Human:", "Assistant:"]


class BasicModelMetadata(BaseModel):
    model_id: str
    quantization: str


class ModelMetadata(BasicModelMetadata):
    conversion_date: str
    parameters: ModelMetadataParameters

    def save(self, output_path: str) -> None:
        """
        Save model metadata to a JSON file.

        Args:
            metadata: The model metadata to save
            output_path: The directory to save the metadata in
        """
        try:
            # Save to JSON file
            metadata_path = os.path.join(output_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.model_dump_json(), f, indent=2)

            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise

    @classmethod
    def load(cls, metadata_path: str):
        """
        Load model metadata from a JSON file.

        Args:
            metadata_path: Path to the metadata JSON file

        Returns:
            The loaded model metadata
        """
        if ModelMetadataFormat.get_format(metadata_path) == ModelMetadataFormat.SIMPLE:
            raise ValueError("Metadata file is not in the correct format (cannot be SIMPLE).")

        try:
            with open(metadata_path, "r") as f:
                return ModelMetadata.model_validate(from_json(json.load(f), allow_partial=True))

        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise


class SimpleModelMetadata(BaseModel):
    """Metadata for a converted model."""
    name: str
    architecture: str = Field(description="Architecture of the model(like Qwen, OPT, etc.)")
    quantization: str = Field(description="Quantization format of the model(like w8a8, opt, hybrid, etc.)")
    quantization_opt: Optional[int] = None
    quantization_hybrid_ratio: Optional[float] = None
    parameters: int = Field(description="Number of parameters in the model (converted to int)")
    context_length: int
    system_prompt: str
    temperature: float
    model_type: Optional[ModelType]


    @staticmethod
    def get_metadata_fields():
        return ["name", "architecture", "quantization", "parameters", "context_length", "system_prompt", "temperature", "model_type"]

    # model_name=Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
    @staticmethod
    def parse_model_name(model_name: str) -> Tuple[dict, dict, List[str]]:
        model_metadata = {}
        model_details = {}
        model_tags: List[str] = []
        splitted = model_name.split("-")
        start_pos = 0

        model_metadata, model_details, model_tags, start_pos = \
            SimpleModelMetadata.parse_splitted_for_model_family(
                splitted=splitted, start_pos=start_pos,
                model_metadata=model_metadata, model_details=model_details, model_tags=model_tags)

        if len(splitted) > start_pos:
            if ModelPath.get_parameter_size(splitted[start_pos]):
                parameters = ModelPath.get_parameter_size(splitted[start_pos])
                model_metadata.update({'parameters': parameters})
                model_tags.append(parameters)
                model_details.update({'parameter_size': parameters})
                start_pos += 1

        if len(splitted) > start_pos:
            # rk3588-1.2.1
            for rk_tag in RK_TAGS_LIST:
                if rk_tag in splitted[start_pos]:
                    founded_rk_tag = rk_tag
                    model_tags.append(rk_tag)
                    model_details.update({'architecture': rk_tag})
                    start_pos += 1
                    if len(splitted) > start_pos+1:
                        lib_vers_match = re.search(r"(\d+\.\d+\.\d+)", splitted[start_pos+1])
                        if lib_vers_match:
                            lib_vers = lib_vers_match.group(1)
                            if lib_vers not in BACKEND_SUPPORTED_LIB_VERSION.get(founded_rk_tag, []):
                                raise ValueError(f"Library version {lib_vers} is not supported for {founded_rk_tag} backend.")
                            else:
                                model_tags.append(lib_vers)
                            start_pos += 1
                    break

        if len(splitted) > start_pos:
            # unsloth-16k
            if re.search(r"(\d+k)", splitted[start_pos]):
                context_length = re.search(r"(\d+k)", splitted[start_pos]).group(1)
                int_context_length = int(context_length[:-1]) * 1024
                model_metadata.update({'context_length': int_context_length})
                model_details.update({'context_length': int_context_length})
                start_pos += 1
            elif 'unsloth' in splitted[start_pos]:
                model_tags.append('unsloth')
                start_pos += 1
                if re.search(r"(\d+k)", splitted[start_pos]):
                    context_length = re.search(r"(\d+k)", splitted[start_pos]).group(1)
                    int_context_length = int(context_length[:-1]) * 1024
                    model_metadata.update({'context_length': int_context_length})
                    model_details.update({'context_length': int_context_length})
                    start_pos += 1

        for rk_tag in RK_TAGS_LIST:
            if rk_tag in splitted and rk_tag not in model_tags:
                model_tags.append(rk_tag)

        if 'architecture' not in model_details:
            for rk_tag in model_tags:
                for platform_processor in PlatformProcessor:
                    if rk_tag == platform_processor.value:
                        model_details.update({'architecture': rk_tag})
                        break

        logger.debug(f"parse_model_name: Model metadata={model_metadata}")
        logger.debug(f"parse_model_name: Model details={model_details}")
        logger.debug(f"parse_model_name: Model tags={model_tags}")
        return model_metadata, model_details, model_tags

    @staticmethod
    def parse_splitted_for_model_family(splitted: List[str], start_pos: int,
            model_metadata: dict, model_details: dict, model_tags: List[str]) -> Tuple[dict, dict, List[str], int]:
        new_pos = start_pos
        if len(splitted) > start_pos:
            if MODEL_SPECS.get(splitted[start_pos].split('.')[0].lower()):
                name = splitted[start_pos].lower()
                model_metadata.update({'name': name})
                model_tags.append(name)
                model_details.update({'model_family': name.split('.')[0]})
                new_pos += 1

        if new_pos == start_pos and len(splitted) > start_pos+1:
            if MODEL_SPECS.get(f"{splitted[start_pos]}-{splitted[start_pos+1]}".lower()):
                name = f"{splitted[start_pos]}-{splitted[start_pos+1]}".lower()
                model_metadata.update({'name': name})
                model_tags.append(name)
                model_details.update({'model_family': name})
                new_pos += 2
            elif MODEL_SPECS.get(f"{splitted[start_pos]}_{splitted[start_pos+1]}".lower()):
                name = f"{splitted[start_pos]}_{splitted[start_pos+1]}".lower()
                model_metadata.update({'name': name})
                model_tags.append(name)
                model_details.update({'model_family': name})
                new_pos += 2

        return model_metadata, model_details, model_tags, new_pos

    # file=Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm,
    # repo=dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k
    @staticmethod
    def parse_file(file: str) -> Tuple[dict, dict, List[str]]:
        model_metadata = {}
        model_details = {}
        model_tags: List[str] = []

        for mtype in ModelType:
            if file.endswith(mtype.get_extension()):
                model_metadata.update({'model_type': mtype})
                file = re.sub(f"{mtype.get_extension()}$", "", file)
                logger.debug(f"parse_file: Model type={mtype}, file reduce to {file}")
                break

        splitted = file.split("-")
        start_pos = 0

        model_metadata, model_details, model_tags, start_pos = \
            SimpleModelMetadata.parse_splitted_for_model_family(
                splitted=splitted, start_pos=start_pos,
                model_metadata=model_metadata, model_details=model_details, model_tags=model_tags)

        if len(splitted) > start_pos:
            if ModelPath.get_parameter_size(splitted[start_pos]):
                parameters = ModelPath.get_parameter_size(splitted[start_pos])
                model_metadata.update({'parameters': parameters})
                model_tags.append(parameters)
                model_details.update({'parameter_size': parameters})
                start_pos += 1

        if len(splitted) > start_pos:
            # rk3588
            for rk_tag in RK_TAGS_LIST:
                if rk_tag in splitted[start_pos]:
                    model_tags.append(rk_tag)
                    for platform_processor in PlatformProcessor:
                        if rk_tag == platform_processor.value:
                            model_details.update({'architecture': rk_tag})
                            break
                    start_pos += 1
                    break

        if len(splitted) > start_pos:
            if splitted[start_pos].lower() in RK_QUANT_FORMAT:
                # w8a8-opt-0-hybrid-ratio-0.0.rkllm
                rk_quant_format = splitted[start_pos].lower()
                start_pos += 1
                if len(splitted) > start_pos:
                    if f"{splitted[start_pos-1]}_{splitted[start_pos]}".lower() in RK_QUANT_FORMAT:
                        rk_quant_format = f"{splitted[start_pos-1]}_{splitted[start_pos]}".lower()
                        start_pos += 1
                model_metadata.update({'quantization': rk_quant_format})
                model_tags.append(rk_quant_format)

                if len(splitted) > start_pos+1:
                    if "opt" == splitted[start_pos] and re.search(r"(\d+)", splitted[start_pos+1]):
                        quant_opt = re.search(r"(\d+)", splitted[start_pos+1]).group(1)
                        model_metadata.update({'quantization_opt': int(quant_opt)})
                        start_pos += 2
                if len(splitted) > start_pos+2:
                    if "hybrid" == splitted[start_pos] and "ratio" == splitted[start_pos+1] and re.search(r"(\d+\.\d+)", splitted[start_pos+2]):
                        quant_hybrid_ratio = re.search(r"(\d+\.\d+)", splitted[start_pos+2]).group(1)
                        model_metadata.update({'quantization_hybrid_ratio': float(quant_hybrid_ratio)})
                        start_pos += 2

        for rk_tag in RK_TAGS_LIST:
            if rk_tag in splitted and rk_tag not in model_tags:
                model_tags.append(rk_tag)

        if 'architecture' not in model_details:
            for rk_tag in model_tags:
                for platform_processor in PlatformProcessor:
                    if rk_tag == platform_processor.value:
                        model_details.update({'architecture': rk_tag})
                        break

        logger.debug(f"parse_file: Model metadata={model_metadata}")
        logger.debug(f"parse_file: Model details={model_details}")
        logger.debug(f"parse_file: Model tags={model_tags}")
        return model_metadata, model_details, model_tags

    @classmethod
    def compute(cls, model_path: ModelPath, model_details: ModelDetails, system_prompt: str, temperature: float = None) -> dict:
        model_metadata_from_name, model_details_from_name, model_tags_from_name = \
            SimpleModelMetadata.parse_model_name(model_path.model_name)

        model_metadata_from_file, model_details_from_file, model_tags_from_file = \
            SimpleModelMetadata.parse_file(model_path.endpoint_model_file)

        model_metadata_from_name.update(model_metadata_from_file)
        model_details_from_name.update(model_details_from_file)
        model_tags_from_name.extend(model_tags_from_file)

        for attr in model_details_from_name:
            try:
                if attr not in model_details.__dict__:
                    model_details.__setattr__(attr, model_details_from_name[attr])
                elif model_details.__dict__[attr] == "Unknown":
                    model_details.__setattr__(attr, model_details_from_name[attr])
            except Exception as e:
                logger.warning(f"Cannot set model details attribute {attr}: {str(e)}")

        model_architecture = model_details.model_family \
                if model_details.model_family is not None \
                else get_model_architecture(model_path.endpoint_model_file)

        if model_architecture is None and 'model_family' in model_details_from_name:
            model_architecture = model_details_from_name['model_family']

        _, _, int_size_value = int_parameters_size(model_metadata_from_name.get('parameters',
                                                                                model_details.parameter_size))

        model_type = model_path.model_type
        if model_type is None and 'model_type' in model_metadata_from_name:
            model_type = model_metadata_from_name['model_type']

        to_return = {
            "name": model_metadata_from_name['name'],
            "architecture": model_architecture,
            "quantization": model_metadata_from_name.get('quantization',
                                                         ollama_quant_mapping.get(model_details.quantization_level)),
            "parameters": int_size_value,
            "context_length": default_context_length(model_architecture),
            "system_prompt": system_prompt,
            "temperature": temperature if temperature is not None \
                else ModelMetadataParameters().temperature,
            "model_type": model_type
        }
        if 'context_length' in model_metadata_from_name:
            to_return.update({'context_length': model_metadata_from_name['context_length']})
        if 'quantization_opt' in model_metadata_from_name:
            to_return.update({'quantization_opt': model_metadata_from_name['quantization_opt']})
        if 'quantization_hybrid_ratio' in model_metadata_from_name:
            to_return.update({'quantization_hybrid_ratio': model_metadata_from_name['quantization_hybrid_ratio']})

        return to_return

    @staticmethod
    def create_using_huggingface_model_info(model_metadata_data: dict, huggingface_model_info: HFModelInfo) -> Any:
        """
        for all fields in HFModelInfo, check if they are existing in the model metadata and if, then update them

        huggingface_model_info = {
            hf_id = None
            id = 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k'
            private = False
            tags = ['qwen3', 'unsloth', 'base_model:Qwen/Qwen3-1.7B', 'base_model:finetune:Qwen/Qwen3-1.7B', 'region:us',
                    'rockchip', 'rk3588']
            downloads = 3
            likes = 1
            modelId = 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k'
            author = 'dulimov'
            sha = 'd2dfbce3a30f60b584734d4e06cf145ace54a578'
            lastModified = '2025-05-24T21:37:33.000Z'
            gated = False
            disabled = False
            model_index = None
            config = HFModelConfig(architectures=['Qwen3ForCausalLM'], model_type='qwen3',
                                   tokenizer_config={'bos_token': None, 'eos_token': '<|im_end|>',
                                                     'pad_token': '<|vision_pad|>', 'unk_token': None},
                                   chat_template_jinja='{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0].role == \'system\' %}\n        {{- messages[0].content + \'\\n\\n\' }}\n    {%- endif %}\n    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0].role == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0].content + \'<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for forward_message in messages %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- set message = messages[index] %}\n    {%- set current_content = message.content if message.content is not none else \'\' %}\n    {%- set tool_start = \'<tool_response>\' %}\n    {%- set tool_start_length = tool_start|length %}\n    {%- set start_of_message = current_content[:tool_start_length] %}\n    {%- set tool_end = \'</tool_response>\' %}\n    {%- set tool_end_length = tool_end|length %}\n    {%- set start_pos = (current_content|length) - tool_end_length %}\n    {%- if start_pos < 0 %}\n        {%- set start_pos = 0 %}\n    {%- endif %}\n    {%- set end_of_message = current_content[start_pos:] %}\n    {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = \'\' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if \'</think>\' in message.content %}\n                {%- set content = (message.content.split(\'</think>\')|last).lstrip(\'\\n\') %}\n                {%- set reasoning_content = (message.content.split(\'</think>\')|first).rstrip(\'\\n\') %}\n                {%- set reasoning_content = (reasoning_content.split(\'<think>\')|last).lstrip(\'\\n\') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- \'<|im_start|>\' + message.role + \'\\n<think>\\n\' + reasoning_content.strip(\'\\n\') + \'\\n</think>\\n\\n\' + content.lstrip(\'\\n\') }}\n            {%- else %}\n                {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- \'\\n\' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- \'<tool_call>\\n{"name": "\' }}\n                {{- tool_call.name }}\n                {{- \'", "arguments": \' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- \'}\\n</tool_call>\' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- message.content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\n\' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- \'<think>\\n\\n</think>\\n\\n\' }}\n    {%- endif %}\n{%- endif %}')
            cardData = HFCardData(base_model=['Qwen/Qwen3-1.7B'], tags=['unsloth'], params=1700000000)
            siblings = [HFSibling(rfilename='.gitattributes'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-1.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.5.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-0.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-0.5.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-1.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.5.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-1.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-0.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-0.5.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-1.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.5.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-1.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.5.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-1.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.0.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.5.rkllm'),
                        HFSibling(rfilename='Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-1.0.rkllm'),
                        HFSibling(rfilename='README.md'), HFSibling(rfilename='added_tokens.json'),
                        HFSibling(rfilename='chat_template.jinja'), HFSibling(rfilename='config.json'),
                        HFSibling(rfilename='generation_config.json'), HFSibling(rfilename='merges.txt'),
                        HFSibling(rfilename='special_tokens_map.json'), HFSibling(rfilename='tokenizer.json'),
                        HFSibling(rfilename='tokenizer_config.json'), HFSibling(rfilename='vocab.json')]
            spaces = []
            createdAt = '2025-05-24T21:26:17.000Z'
            usedStorage = 17308806896
            languages = ['en']
            }

        converted to:
            name: str
            architecture: str
            quantization: str
            quantization_opt: Optional[int] = None
            quantization_hybrid_ratio: Optional[float] = None
            parameters: int
            context_length: int
            system_prompt: str
            temperature: float
            model_type: Optional[ModelType]

        """

        # architecture
        if huggingface_model_info.config.model_type:
            if MODEL_SPECS.get(huggingface_model_info.config.model_type):
                model_metadata_data['architecture'] = huggingface_model_info.config.model_type
            else:
                raise Exception(f"model type {huggingface_model_info.config.model_type} not yet supported")
        elif 'architecture' not in model_metadata_data:
            # search from tag
            for tag in huggingface_model_info.tags:
                if tag in MODEL_SPECS:
                    model_metadata_data['architecture'] = tag
                    break

        # parameters
        if huggingface_model_info.cardData:
            if huggingface_model_info.cardData.params > 0:
                model_metadata_data['parameters'] = huggingface_model_info.cardData.params

        logger.debug(f"model_metadata_data: {model_metadata_data}")

        return SimpleModelMetadata(**model_metadata_data)

    @staticmethod
    def create_using_ollama_model_info(model_metadata_data: dict, ollama_model_info: OllamaModelInfo) -> Any:
        """
        for all fields in OllamaModelInfo, check if they are existing in the model metadata and if, then update them
        ollama_model_info = {
            model_format = 'gguf'
            model_family = 'qwen2'
            model_families = ['qwen2']
            model_type = '494.03M'
            file_type = 'Q4_K_M'
            architecture = 'amd64'
            os = 'linux'
            rootfs = OllamaRootfs(type='layers',
                                  diff_ids=['sha256:c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515',
                                            'sha256:75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de',
                                            'sha256:9bebd78bf5bc92d41d5f3aab3ee66c891376b4eb4cf433edc2533c2f5f9c95a6',
                                            'sha256:832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e'])
            }

        converted to:
            name: str
            architecture: str
            quantization: str
            quantization_opt: Optional[int] = None
            quantization_hybrid_ratio: Optional[float] = None
            parameters: int
            context_length: int
            system_prompt: str
            temperature: float
            model_type: Optional[ModelType]
        """
        # model_family = 'qwen2'
        # architecture
        if ollama_model_info.model_family:
            if MODEL_SPECS.get(ollama_model_info.model_family):
                model_metadata_data['architecture'] = ollama_model_info.model_family
            else:
                raise Exception(f"model type {ollama_model_info.model_type} not yet supported")

        # file_type = 'Q4_K_M'
        # quantization: str
        if ollama_model_info.file_type:
            if ollama_model_info.file_type in OLLAMA_QUANT_FORMAT:
                model_metadata_data['quantization'] = ollama_model_info.file_type
            else:
                raise Exception(f"model quantization {ollama_model_info.file_type} not yet supported")

        # model_type = '494.03M'
        # parameters: int
        if ollama_model_info.model_type:
            size_value, size_unit, int_size_value = int_parameters_size(ModelPath.get_parameter_size(ollama_model_info.model_type))
            if ollama_model_info.model_type.startswith(str(size_value)):
                model_metadata_data['parameters'] = int_size_value

        # model_format = 'gguf'
        # model_type: Optional[ModelType]
        if ollama_model_info.model_format:
            for mt in ModelType:
                if mt.value.lower() == ollama_model_info.model_format.lower():
                    model_metadata_data['model_type'] = mt
                    break

        logger.debug(f"model_metadata_data: {model_metadata_data}")

        return SimpleModelMetadata(**model_metadata_data)

    @classmethod
    def from_complete(cls, metadata: ModelMetadata):
        data: dict = {}
        for attr in metadata.__dict__:
            if attr == "model_id":
                data["name"] = metadata.__dict__[attr]
            elif attr == "parameters":
                for param_attr in metadata.parameters.__dict__:
                    data[param_attr] = metadata.parameters.__dict__[param_attr]

        return cls(**data)

    def save(self, output_path: str) -> None:
        """
        Save model metadata to a JSON file.

        Args:
            metadata: The model metadata to save
            output_path: The directory to save the metadata in
        """
        try:
            # Save to JSON file
            metadata_path = os.path.join(output_path, METADATA_FILENAME)
            with open(metadata_path, "w") as f:
                json.dump(self.model_dump_json(), f, indent=2)

            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise

    @classmethod
    def load(cls, metadata_path: str):
        """
        Load model metadata from a JSON file.

        Args:
            metadata_path: Path to the metadata JSON file

        Returns:
            The loaded model metadata
        """
        if ModelMetadataFormat.get_format(metadata_path) != ModelMetadataFormat.SIMPLE:
            raise ValueError(
                f"Metadata file is not in the correct format (search for SIMPLE, is {ModelMetadataFormat.get_format(metadata_path)}).")

        try:
            with open(metadata_path, "r") as f:
                return SimpleModelMetadata.model_validate(from_json(json.load(f), allow_partial=True))

        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise


def get_model_size(model_path: str) -> int:
    """Get the size of a model file in bytes."""
    return os.path.getsize(model_path)


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_model_architecture(model_path: str) -> Optional[str]:
    """Detect the model architecture from the model file."""
    # TODO: Implement architecture detection
    pass
