from fastapi import APIRouter

from core.api.parameters.converter.ConversionConfig import ConversionConfig
from core.model.converter.HuggingFaceToRKLLMConverter import HuggingFaceToRKLLMConverter

router = APIRouter(tags=["converter"])


@router.post("/convert/rkllm")
async def convert_rkllm(config: ConversionConfig):
    try:
        # Create converter and run conversion
        converter = HuggingFaceToRKLLMConverter(config)
        converter.convert()

        return 0

    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        return 1
