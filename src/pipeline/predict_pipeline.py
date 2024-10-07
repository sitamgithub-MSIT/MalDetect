# Importing required libraries
import sys
from PIL import Image
import numpy as np
import keras
from keras.utils import img_to_array

# Local imports
from src.components.optimizer import WarmUpCosineDecay
from src.logger import logging
from src.exception import CustomExceptionHandling
from src.config import IMAGE_SIZE, MODEL_PATH


# Class labels
CLASS_LABELS = {
    0: "Parasitized",
    1: "Uninfected",
}


def load_model(MODEL_PATH: str) -> keras.Model:
    """
    Load the trained model from the given path.

    Args:
        MODEL_PATH (str): The path to the trained model.

    Returns:
        keras.Model: The trained model.
    """
    # Loading model
    try:
        return keras.saving.load_model(
            MODEL_PATH, custom_objects={"WarmUpCosineDecay": WarmUpCosineDecay}
        )
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e


model = load_model(MODEL_PATH=MODEL_PATH)


def preprocess_img(img_path: str) -> np.ndarray:
    """
    Preprocesses an image by opening it, resizing it to IMAGE_SIZE, converting it to an array,
    and normalizing the pixel values to the range [0, 1].

    Args:
        img_path (str): The path to the image file.

    Returns:
        np.ndarray: The preprocessed image as a 4D array with shape (1, 224, 224, 3).
    """
    try:
        # Opening the image file using PIL Image
        op_img = Image.open(img_path)
        logging.info("Image opened successfully.")

        # Resizing the image to IMAGE_SIZE and converting it to an array
        img_resize = op_img.resize(IMAGE_SIZE)
        img2arr = img_to_array(img_resize)

        # Reshaping the image to (1, 224, 224, 3)
        logging.info("Image preprocessed successfully.")
        return img2arr.reshape(1, *IMAGE_SIZE, 3)

    # Handle exceptions that may occur during image processing
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e


def predict_result(predict: np.ndarray) -> str:
    """
    Predicts the result for the given input.

    Args:
        predict (np.ndarray): The input data for prediction.

    Returns:
        str: The predicted result.
    """
    try:
        # Predicting the result
        pred = model.predict(predict)
        result = CLASS_LABELS[np.argmax(pred[0], axis=-1)]
        logging.info("Prediction successful.")
        return result

    # Handle exceptions that may occur during prediction
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e
