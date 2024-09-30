# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Loading model
model = load_model("artifacts\model.h5")


# Preparing and pre-processing the image
def preprocess_img(img_path):
    """
    Preprocesses an image by opening it, resizing it to (224, 224), converting it to an array,
    and normalizing the pixel values to the range [0, 1].

    Args:
        img_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The preprocessed image as a 4D array with shape (1, 224, 224, 3).
    """
    # Opening the image file using PIL Image
    op_img = Image.open(img_path)

    # Resizing the image to (224, 224) and converting it to an array
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize)

    # Reshaping the image to (1, 224, 224, 3)
    return img2arr.reshape(1, 224, 224, 3)


# Predicting function
def predict_result(predict):
    """
    Predicts the result for the given input.

    Args:
        predict (numpy.ndarray): The input data for prediction.

    Returns:
        int: The predicted result.
    """
    # Predicting the result
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
