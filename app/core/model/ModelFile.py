import os
import time

from dotenv import load_dotenv
from pydantic import BaseModel

from core import config
from src import variables as variables, RKLLM
from src.model_utils import get_model_full_options
from ui import print_color


class ModelFile(BaseModel):
    file: str


# TODO: move as static method "create" in ModelFile
def create_modelfile(huggingface_path, From, system="", model_name=None):
    struct_modelfile = f"""
FROM="{From}"

HUGGINGFACE_PATH="{huggingface_path}"

SYSTEM="{system}"

TEMPERATURE={config.get("model", "default_temperature")}

ENABLE_THINKING={config.get("model", "default_enable_thinking")}

NUM_CTX={config.get("model", "default_num_ctx")}

MAX_NEW_TOKENS={config.get("model", "default_max_new_tokens")}

TOP_K={config.get("model", "default_top_k")}

TOP_P={config.get("model", "default_top_p")}

REPEAT_PENALTY={config.get("model", "default_repeat_penalty")}

FREQUENCY_PENALTY={config.get("model", "default_frequency_penalty")}

PRESENCE_PENALTY={config.get("model", "default_presence_penalty")}

MIROSTAT={config.get("model", "default_mirostat")}

MIROSTAT_TAU={config.get("model", "default_mirostat_tau")}

MIROSTAT_ETA={config.get("model", "default_mirostat_eta")}


"""

    # Use config for models path
    # path = os.path.join(config.get_path("models"), From.replace('.rkllm', ''))
    path = os.path.join(config.get_path("models"), model_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Create the Modelfile and write the content
    with open(os.path.join(path, "Modelfile"), "w") as f:
        f.write(struct_modelfile)


# TODO: move as method "load" in ModelFile, that returns a Model object
def load_model(
    model_name, huggingface_path=None, system="", From=None, request_options=None
):
    # Use config for models path
    model_dir = os.path.join(config.get_path("models"), model_name)

    if not os.path.exists(model_dir):
        return None, f"Model directory '{model_name}' not found."

    if not os.path.exists(os.path.join(model_dir, "Modelfile")) and (
        huggingface_path is None and From is None
    ):
        return None, f"Modelfile not found in '{model_name}' directory."
    elif huggingface_path is not None and From is not None:
        create_modelfile(
            huggingface_path=huggingface_path,
            From=From,
            system=system,
            model_name=model_name,
        )
        time.sleep(0.1)

    # Load modelfile
    load_dotenv(os.path.join(model_dir, "Modelfile"), override=True)

    from_value = os.getenv("FROM")
    huggingface_path = os.getenv("HUGGINGFACE_PATH")

    # View config Vars
    print_color(f"FROM: {from_value}\nHuggingFace Path: {huggingface_path}", "green")

    if not from_value or not huggingface_path:
        return None, "FROM or HUGGINGFACE_PATH not defined in Modelfile."

    # Change value of model_id with huggingface_path
    variables.model_id = huggingface_path

    # Get model parameters if not provided
    if not request_options:
        request_options = get_model_full_options(
            model_name, config.get_path("models"), request_options
        )

    try:
        modele_rkllm = RKLLM(
            os.path.join(model_dir, from_value), model_dir, options=request_options
        )
    except RuntimeError as e:
        return None, str(e)

    return modele_rkllm, None
