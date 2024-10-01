import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import necessary libraries
import keras
import keras_cv
import tensorflow as tf
from keras import optimizers
from keras import losses

# Local imports
from src.components.dataset import load_and_prepare_dataset
from src.components.augmentation import get_augmenters
from src.components.optimizer import WarmUpCosineDecay
from src.components.model import build_model
from src.utils import create_augmenter_fn, unpackage_dict
from src.config import (
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    SHUFFLE_BUFFER_MULTIPLIER,
    DATASET_NAME,
    BATCH_SIZE,
    EPOCHS,
    IMAGE_SIZE,
    TOP_DROPOUT_RATE,
    START_LR,
    TARGET_LR,
    WEIGHT_DECAY,
    MOMENTUM,
    PATIENCE,
    LABEL_SMOOTHING,
    NUM_CLASSES,
)
from src.logger import logging

# Set the seeds for reproducibility
SEEDS = 42
keras.utils.set_random_seed(SEEDS)

# Set the total number of images and steps for warmup and hold
total_images = 27558
total_steps = (total_images // BATCH_SIZE) * EPOCHS
warmup_steps = int(0.1 * total_steps)
hold_steps = int(0.45 * total_steps)

# Load and prepare the dataset
train_ds, eval_ds, test_ds = load_and_prepare_dataset(
    dataset_name=DATASET_NAME,
    batch_size=BATCH_SIZE,
    TRAIN_SPLIT=TRAIN_SPLIT,
    VAL_SPLIT=VAL_SPLIT,
    TEST_SPLIT=TEST_SPLIT,
    SHUFFLE_BUFFER_MULTIPLIER=SHUFFLE_BUFFER_MULTIPLIER,
)

# Get the augmenters and apply them to the train set
augmenters = get_augmenters()
augmenter_fn = create_augmenter_fn(augmenters)
train_ds = train_ds.map(augmenter_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Resizing the images in the train and test set and eval set
inference_resizing = keras_cv.layers.Resizing(
    IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True
)
train_ds = train_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

# Unpackage the dictionaries in the train and test sets
train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

# Define the learning rate schedule
schedule = WarmUpCosineDecay(
    start_lr=START_LR,
    target_lr=TARGET_LR,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    hold=hold_steps,
)

# Optimizer function for the model
optimizer_fn = optimizers.SGD(
    weight_decay=WEIGHT_DECAY,
    learning_rate=schedule,
    momentum=MOMENTUM,
)

# Loss function for the model
loss_fn = losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)

# Callbacks for the training process
train_callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=f"./artifacts/{DATASET_NAME}_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    ),
    keras.callbacks.TensorBoard(log_dir="./tensorboard_logs"),
]

# Model summary
model = build_model(NUM_CLASSES, IMAGE_SIZE, optimizer_fn, loss_fn, TOP_DROPOUT_RATE)

# Model training
history = model.fit(
    train_ds, epochs=EPOCHS, callbacks=train_callbacks, validation_data=eval_ds
)
logging.info(
    f"Model training completed successfully after {EPOCHS} epochs with {history.history}"
)

# Evaluate the model on the test set
accuracy = model.evaluate(test_ds)[1] * 100
logging.info(f"Model accuracy on the test set: {accuracy:.2f}%")
