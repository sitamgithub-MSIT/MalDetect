# Import libraries
import sys
from typing import Tuple
import keras
from keras import layers
from keras.applications import EfficientNetB0
from keras.metrics import (
    BinaryAccuracy,
    Precision,
    Recall,
    AUC,
    FalsePositives,
    FalseNegatives,
    TruePositives,
    TrueNegatives,
)

# Local imports
from src.logger import logging
from src.exception import CustomExceptionHandling


def build_model(
    num_classes: int,
    image_size: Tuple[int, int],
    optimizer_fn: keras.optimizers.Optimizer,
    loss_fn: keras.losses.Loss,
    top_dropout_rate: float,
) -> keras.Model:
    """
    Build a model for malaria diagnosis using EfficientNet.

    Args:
        - num_classes (int): The number of classes for classification.
        - image_size (Tuple[int, int]): The size of the input image.
        - optimizer_fn (keras.optimizers.Optimizer): The optimizer to use for training.
        - loss_fn (keras.losses.Loss): The loss function to use for training.
        - top_dropout_rate (float): The dropout rate for the top layers.

    Returns:
        keras.Model: The compiled model.
    """
    try:
        logging.info("Starting to build the model...")

        # Create the base model from the pre-trained model EfficientNet
        inputs = layers.Input(shape=(image_size[0], image_size[1], 3))
        base_model = EfficientNetB0(
            include_top=False, input_tensor=inputs, weights="imagenet"
        )

        # Freeze the pretrained weights
        base_model.trainable = False

        # Rebuild top of pretrained model
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(num_classes, activation="sigmoid", name="pred")(x)

        # Compile the model
        model = keras.Model(inputs, outputs, name="EfficientNet")
        model.compile(
            optimizer=optimizer_fn,
            loss=loss_fn,
            metrics=[
                BinaryAccuracy(name="accuracy"),
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc"),
                FalsePositives(name="false_positives"),
                FalseNegatives(name="false_negatives"),
                TruePositives(name="true_positives"),
                TrueNegatives(name="true_negatives"),
            ],
        )

        logging.info("Model built successfully.")
        return model

    # Handle exceptions that may occur during model building
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e
