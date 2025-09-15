import json
import os
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

from core.api.parameters.converter.ConversionConfig import ConversionConfig
from core.model.Model import Model
from core.model.ModelInfo import ModelDetails
from core.model.ModelPath import ModelPath
from core.model.ModelType import ModelType
from core.model.converter.RKLLMConverter import RKLLMConverter, RKLLMConverterConfig
from core.model.converter.quantization import QuantizationConverter
from core.model.converter import logger, quantization_constants


class HuggingFaceToRKLLMConverter:
    """Converts Hugging Face models to RKLLM format."""

    OLLAMA_QUANTIZATION_MAPPING = quantization_constants.ollama_quant_mapping

    MODEL_TYPE: ModelType = ModelType.RKLLM

    @property
    def model_type(self) -> ModelType:
        return self.MODEL_TYPE

    def __init__(self, config: ConversionConfig):
        self.config = config
        self._validate_config()
        self.model = None
        self.metadata = None
        self.tokenizer = None
        self.processor = None
        self.rkllm_converter = None

    def _validate_config(self) -> None:
        """Validate the conversion configuration."""
        if not self.config.quantization in self.OLLAMA_QUANTIZATION_MAPPING:
            raise ValueError(f"Unsupported quantization: {self.config.quantization}")

    def convert(self) -> None:
        """Main conversion method."""
        logger.info(f"Starting conversion of {self.config.model_name}")

        # TODO: fulfill model_details using self.config and model_id
        model_details: ModelDetails =ModelDetails(**{
            'format': '?'
        })

        endpoint_model_file: str = ModelPath.gen_endpoint_model_file_name_using_model_details(
            model_name=self.config.model_name,
            model_type=self.model_type,
            model_details=model_details
        )

        model_path: ModelPath = ModelPath(**{
            'model_name': self.config.model_name,
            'model_type': self.model_type,
            'huggingface_path': self.config.model_id,
            'endpoint_model_file': endpoint_model_file
        })

        lock_id = model_path.lock_model()
        if lock_id >= 0:
            # Step 1: Load model and tokenizer
            self._load_model_and_tokenizer()

            # Step 2: Convert weights
            self._convert_weights()

            # Step 3: Generate RKLLM file
            self._generate_rkllm_file()

            # Step 4: Create Modelfile
            self._create_modelfile()

            # Step 5: Save metadata
            self._save_metadata(self.config.output_path)

            logger.info("Conversion completed successfully")

            model_path.unlock_model(lock_id)
        else:
            logger.info("Model is already being converted, skipping conversion")
            return

    def _load_model_and_tokenizer(self) -> None:
        """Load the model and tokenizer from Hugging Face."""
        logger.info("Loading model and tokenizer...")
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                token=self.config.token
            )

            # Try to load processor for multimodal models
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.config.model_id,
                    token=self.config.token
                )
                logger.info("Loaded multimodal processor")
            except:
                logger.info("No multimodal processor found, using text-only mode")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                token=self.config.token,
                torch_dtype=torch.float16 if self.config.dtype == 'float16' else torch.float32,
                device_map=self.config.device
            )

            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {str(e)}")
            raise

    def _convert_weights(self) -> None:
        """Convert model weights to RKLLM format."""
        logger.info("Converting weights...")
        try:
            # Get the target quantization format
            target_format = self.OLLAMA_QUANTIZATION_MAPPING[self.config.quantization]

            # Convert weights using our quantization converter
            self.model = QuantizationConverter.convert_weights(
                self.model,
                self.config.quantization,
                target_format
            )
            logger.info("Weights converted successfully")
        except Exception as e:
            logger.error(f"Error converting weights: {str(e)}")
            raise

    def _generate_rkllm_file(self, endpoint_model_file: str) -> None:
        """Generate the RKLLM binary file."""
        logger.info("Generating RKLLM file...")
        try:
            # Initialize RKLLM converter
            self.rkllm_converter = RKLLMConverter(
                model=self.model,
                config=RKLLMConverterConfig(**{
                    'quantization': self.config.quantization,
                    'max_context_len': self.config.max_context_len
                })
            )

            # Convert and save RKLLM file with model name
            output_path = os.path.join(self.config.output_path, f'{endpoint_model_file}{self.MODEL_TYPE.get_extension()}')
            self.rkllm_converter.convert(output_path)

            logger.info(f"RKLLM file generated at {output_path}")
        except Exception as e:
            logger.error(f"Error generating RKLLM file: {str(e)}")
            raise

    def _create_modelfile(self) -> None:
        """Create Modelfile for the converted model."""
        logger.info("Creating Modelfile...")

        # Extract model name from model_id
        model_name = self.config.model_id.split('/')[-1]

        modelfile_content = f"""FROM="{model_name}.rkllm"
HUGGINGFACE_PATH="{self.config.model_id}"
SYSTEM="You are a helpful AI assistant."
TEMPERATURE=0.7
"""

        modelfile_path = os.path.join(self.config.output_path, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        logger.info(f"Modelfile created at {modelfile_path}")

    def _save_metadata(self, output_dir: str) -> None:
        """Save metadata about the conversion to a JSON file."""
        metadata = {
            "model_id": self.config.model_id,
            "quantization": self.config.quantization,
            "conversion_date": datetime.now().isoformat(),
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
                "stop_sequences": ["Human:", "Assistant:"]
            }
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")
