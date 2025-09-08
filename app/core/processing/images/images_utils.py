import base64
import os

import numpy as np
import requests

from core import config
from src.format_utils import logger


def expand_to_square(img, background_color=(127.5, 127.5, 127.5)):
    """Expand an image to a square by adding borders with the specified background color.
    Args:
        img: Input image as a numpy array (HWC).
        background_color: Tuple of 3 values for BGR background color.
    Returns:
        Squared image as a numpy array (HWC).
    """
    h, w = img.shape[:2]
    if h == w:
        return img.copy()
    size = max(h, w)
    # OpenCV C++ saturate_cast<uchar> rounds to nearest for positive values:
    bg = tuple(int(np.rint(v)) for v in background_color)  # 127.5 -> 128

    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left

    return cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=bg
    )


def prepare_image(image_path, width, height) -> np.ndarray:
    """ Load and preprocess an image for model input.
        Args:
            image_path: Path, URL, or Base64 string of the image.
            width: Target width.
            height: Target height.
        Returns:
            Preprocessed image as a numpy array (HWC, uint8).
    """
    # Read image
    img = load_image(image_path)  # BGR
    if img is None:
        raise FileNotFoundError(image_path)

    # Preprocess Image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    square = expand_to_square(img, background_color=(127.5, 127.5, 127.5))
    resized = cv2.resize(square, (width, height), interpolation=cv2.INTER_LINEAR)
    if resized.dtype != np.uint8:
        resized = resized.astype(np.uint8)

    resized = np.ascontiguousarray(resized, dtype=np.uint8)
    return resized


def load_image(source: str):
    """
    Load an image from:
      - a local path
      - a URL
      - a Base64 string
    Returns:
      - image as numpy array (BGR) or None if fails
    """
    img = None

    # Case 1: local file
    if os.path.exists(source):
        img = cv2.imread(source)

    # Case 2: URL
    elif source.startswith("http://") or source.startswith("https://"):
        try:
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error("Error loading from URL:", e)

    # Case 3: Base64
    else:
        try:
            # Remove "data:image/..;base64," if present
            if "," in source:
                source = source.split(",")[1]
            img_data = base64.b64decode(source)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error("Error loading from Base64:", e)

    return img


def read_data_from_file(path: str) -> bytes:
    """
    Read binary data from a file.
    Args:
       path: The path to the file
    Returns:
       The binary data read from the file
    """
    # Ensure the file exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Read and return the binary data
    with open(path, "rb") as f:
        return f.read()


def get_encoder_model_path(model_name: str) -> Union[str, None]:
    """
    Get the path of the vision encoder model if exists.

    Args:
        model_name: The name of the model directory

    Returns:
        The path to the vision encoder model or None if not found
    """
    # Get the models directory
    models_dir = config.get_path("models")
    model_path = os.path.join(models_dir, model_name)

    # check for the RKNN file
    encoder_filename = None
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith(".rknn"):
                size = os.path.getsize(os.path.join(model_path, file))
                encoder_filename = file
                break

    # Return the full path if found
    if encoder_filename:
        return os.path.join(model_path, encoder_filename)
    else:
        return None
