# Import tensorflow datasets
import sys
import tensorflow as tf
import tensorflow_datasets as tfds

# Local imports
from src.utils import package_inputs
from src.logger import logging
from src.exception import CustomExceptionHandling

tfds.disable_progress_bar()


def load_and_prepare_dataset(
    dataset_name: str,
    batch_size: int,
    TRAIN_SPLIT: str,
    VAL_SPLIT: str,
    TEST_SPLIT: str,
    SHUFFLE_BUFFER_MULTIPLIER: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads and prepares a dataset for training, validation, and testing.

    Args:
        - dataset_name (str): The name of the dataset to load.
        - batch_size (int): The size of the batches to use for training and evaluation.
        - TRAIN_SPLIT (str): The split of the dataset to use for training.
        - VAL_SPLIT (str): The split of the dataset to use for validation.
        - TEST_SPLIT (str): The split of the dataset to use for testing.
        - SHUFFLE_BUFFER_MULTIPLIER (int): The multiplier for the shuffle buffer size.

    Returns:
        tuple: A tuple containing:
            - train_ds (tf.data.Dataset): The training dataset.
            - eval_ds (tf.data.Dataset): The validation dataset.
            - test_ds (tf.data.Dataset): The testing dataset.
    """
    logging.info(f"Loading dataset: {dataset_name}")
    try:
        # Load the dataset and split it into train, test, and validation sets
        (train_ds, val_ds, test_ds), _ = tfds.load(
            dataset_name,
            split=[TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT],
            with_info=True,
            as_supervised=True,
        )
        logging.info("Dataset loaded successfully")

    # Handle exceptions that may occur during dataset loading
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e

    # Map the `package_inputs` function to the train and test sets
    train_ds = train_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = val_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle the training dataset
    train_ds = train_ds.shuffle(batch_size * SHUFFLE_BUFFER_MULTIPLIER)
    logging.info("Shuffled the training dataset")

    # Applying ragged batching to the train and test sets
    train_ds = train_ds.ragged_batch(batch_size)
    eval_ds = eval_ds.ragged_batch(batch_size)
    test_ds = test_ds.ragged_batch(batch_size)
    logging.info("Applied ragged batching to datasets")

    return train_ds, eval_ds, test_ds
