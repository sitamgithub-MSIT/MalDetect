# Importing keras_cv for applying augmentations
import sys
import keras_cv

# Local imports
from src.logger import logging
from src.exception import CustomExceptionHandling


def get_augmenters():
    """
    Creates a list of augmenters to be applied to the input training data.

    Returns:
        List: List of augmentation layers.
    """
    try:
        logging.info("Creating augmenter functions.")

        # Randomly flip the images in the batch
        random_flip = keras_cv.layers.RandomFlip()
        logging.info("RandomFlip augmenter created.")
        augmenters = [random_flip]

        # Randomly rotate the images in the batch
        random_rotation = keras_cv.layers.RandomRotation(factor=0.15)
        logging.info("RandomRotation augmenter created.")
        augmenters.append(random_rotation)

        # Randomly translate the images in the batch
        random_translation = keras_cv.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1
        )
        logging.info("RandomTranslation augmenter created.")
        augmenters.append(random_translation)

        # Randomly change contrast of the images in the batch
        random_contrast = keras_cv.layers.RandomContrast(
            factor=0.1, value_range=(0, 255)
        )
        logging.info("RandomContrast augmenter created.")
        augmenters.append(random_contrast)

        logging.info("All augmenters created successfully.")
        return augmenters

    # Handle exceptions that may occur during augmentation creation
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e
