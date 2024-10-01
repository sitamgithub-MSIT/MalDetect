import sys
import tensorflow as tf
from typing import Dict, Tuple

# Local imports
from src.config import NUM_CLASSES
from src.logger import logging
from src.exception import CustomExceptionHandling


def package_inputs(image: tf.Tensor, label: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Packages the input image and label into a dictionary.

    Args:
        - image (tf.Tensor): The input image.
        - label (tf.Tensor): The corresponding label.

    Returns:
        Dict[str, tf.Tensor]: A dictionary containing the input image and one-hot encoded label.
    """
    try:
        logging.info("Packaging inputs.")
        packaged = {"images": image, "labels": tf.one_hot(label, NUM_CLASSES)}
        logging.info("Packaging successful.")
        return packaged

    # Handle exceptions that may occur during packaging
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e


def unpackage_dict(inputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Unpackages a dictionary and returns the values of the 'images' and 'labels' keys.

    Args:
        inputs (Dict[str, tf.Tensor]): A dictionary containing the 'images' and 'labels' keys.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the values of the 'images' and 'labels' keys.
    """
    try:
        logging.info("Unpackaging dictionary.")
        images, labels = inputs["images"], inputs["labels"]
        logging.info("Unpackaging successful.")
        return images, labels

    # Handle exceptions that may occur during unpackaging
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e


def create_augmenter_fn(augmenters):
    """
    Creates an augmenter function that applies a list of augmenters to the inputs.

    Args:
        augmenters (List[Callable[[tf.Tensor], tf.Tensor]]): List of augmenter functions to be applied.

    Returns:
        Callable[[tf.Tensor], tf.Tensor]: Augmenter function that applies the augmenters to the inputs.
    """

    def augmenter_fn(inputs):
        """
        Apply a series of augmentations to the input data.

        Args:
            inputs (tf.Tensor): The input data to be augmented.

        Returns:
            tf.Tensor: The augmented data.
        """
        try:
            logging.info("Applying augmentations.")
            for augmenter in augmenters:
                inputs = augmenter(inputs)
            logging.info("Augmentations applied successfully.")
            return inputs

        # Handle exceptions that may occur during augmentation
        except Exception as e:
            # Custom exception handling
            raise CustomExceptionHandling(e, sys) from e

    return augmenter_fn
