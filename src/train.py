"""
Title: Malaria Diagnosis using EfficientNet
Authors: [Sitam Meur](https://github.com/sitamgithub-MSIT)
Date created: 2024/01/24
Last modified: 2024/01/26
Description: Train a model to diagnose malaria using EfficientNet by transfer learning from pre-trained weights.
Accelerator: GPU
"""


"""
# Model Training

## Introduction: what is EfficientNet?

EfficientNet, first introduced in [Tan and Le, 2019](https://arxiv.org/abs/1905.11946) is among the most efficient models (i.e. requiring least FLOPS for inference) that reaches State-of-the-Art accuracy on both imagenet and common image classification transfer learning tasks.

The smallest base model is similar to [MnasNet](https://arxiv.org/abs/1807.11626), which reached near-SOTA with a significantly smaller model. By introducing a heuristic way to scale the model, EfficientNet provides a family of models (B0 to B7) that represents a good combination of efficiency and accuracy on a variety of scales. Such a scaling heuristics (compound-scaling, details see [Tan and Le, 2019](https://arxiv.org/abs/1905.11946)) allows the
efficiency-oriented base model (B0) to surpass models at every scale, while avoiding extensive grid-search of hyperparameters.

## Setup and data loading
"""

# Import numpy, matplotlib, seaborn, and other necessary libraries
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import tensorflow for [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data)
import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

# Import keras and its modules
import keras_cv

import keras
from keras import layers
from keras.applications import EfficientNetB0
from keras import optimizers
from keras import losses
from keras import metrics

# Set the seeds for reproducibility
SEEDS = 42
np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)

"""
### Define Hyperparameters

First, let's define the hyperparameters. We will use a batch size of 32 and train for 10 epochs. We will use the RMSProp optimizer with a learning rate of 0.001. The image size is set to 224x224.
"""

# Set the batch size, image size, and number of epochs
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 10
lr = 1e-4

"""
### Gather Malaria Dataset

Here we load data from [tensorflow_datasets](https://www.tensorflow.org/datasets). Malaria dataset is provided in TFDS as [malaria](https://www.tensorflow.org/datasets/catalog/malaria). It features 27,558 images that belong to 2 classes: parasitized and uninfected.
"""

# Set the dataset name
dataset_name = "malaria"

# Load the dataset and split it into train and test sets
(train_ds, test_ds), ds_info = tfds.load(
    dataset_name,
    split=["train[:80%]", "train[80%:]"],
    with_info=True,
    as_supervised=True,
)

# Get the number of classes in the dataset
NUM_CLASSES = ds_info.features["label"].num_classes


def package_inputs(image, label):
    """
    Packages the input image and label into a dictionary.

    Args:
        image: The input image.
        label: The corresponding label.

    Returns:
        A dictionary containing the input image and label.
    """
    return {"images": image, "labels": tf.one_hot(label, NUM_CLASSES)}


# Map the `package_inputs` function to the train and test sets
train_ds = train_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = test_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle the training dataset
train_ds = train_ds.shuffle(BATCH_SIZE * 16)

"""
### Visualise the dataset

The following code shows the images from the training set.
"""

# Applying raggad batching to the train and test sets
train_ds = train_ds.ragged_batch(BATCH_SIZE)
eval_ds = eval_ds.ragged_batch(BATCH_SIZE)

# Image batch for visualization from the train set
batch = next(iter(train_ds.take(1)))
image_batch = batch["images"]

# Visualize the images in the batch
keras_cv.visualization.plot_image_gallery(
    image_batch.to_tensor(),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)

"""
### Data Augmentation

We can use the keras cv layers to perform data augmentation. In this notebook, we apply various augmentation techniques such as random flipping, rotation, translation, and contrast/brightness adjustment to the input images. These techniques help in increasing the diversity of the training data and improve the model's ability to generalize to unseen examples.
"""

# Randomly flip the images in the batch
random_flip = keras_cv.layers.RandomFlip()
augmenters = [random_flip]

# Apply the augmenters to the image batch
image_batch = random_flip(image_batch)

# Visualize the images in the batch
keras_cv.visualization.plot_image_gallery(
    image_batch.to_tensor(),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)

# Randomly rotate the images in the batch
random_rotation = keras_cv.layers.RandomRotation(factor=0.15)
augmenters = [random_rotation]

# Apply the augmenters to the image batch
image_batch = random_rotation(image_batch)

# Visualize the images in the batch
keras_cv.visualization.plot_image_gallery(
    image_batch.to_tensor(),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)

# Randomly translate the images in the batch
random_translation = keras_cv.layers.RandomTranslation(
    height_factor=0.1, width_factor=0.1
)
augmenters = [random_translation]

# Apply the augmenters to the image batch
image_batch = random_translation(image_batch)

# Visualize the images in the batch
keras_cv.visualization.plot_image_gallery(
    image_batch.to_tensor(),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)

# Randomly change contrast of the images in the batch
random_contrast = keras_cv.layers.RandomContrast(factor=0.1, value_range=(0, 255))
augmenters = [random_contrast]

# Apply the augmenters to the image batch
image_batch = random_contrast(image_batch)

# Visualize the images in the batch
keras_cv.visualization.plot_image_gallery(
    image_batch.to_tensor(),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)


def create_augmenter_fn(augmenters):
    """
    Creates an augmenter function that applies a list of augmenters to the inputs.

    Args:
        augmenters (list): List of augmenter functions to be applied.

    Returns:
        augmenter_fn (function): Augmenter function that applies the augmenters to the inputs.
    """

    def augmenter_fn(inputs):
        """
        Apply a series of augmentations to the input data.

        Args:
            inputs: The input data to be augmented.

        Returns:
            The augmented data.
        """
        for augmenter in augmenters:
            inputs = augmenter(inputs)
        return inputs

    return augmenter_fn


# Apply all the previously defined augmenters to the train set
augmenter_fn = create_augmenter_fn(augmenters)
train_ds = train_ds.map(augmenter_fn, num_parallel_calls=tf.data.AUTOTUNE)

"""
### Prepare Inputs

Once we verify the input data and augmentation are working correctly, we prepare the dataset for training. The input data is resized to a uniform size of 224x224.
"""

# Resizing the images in the train and test sets
inference_resizing = keras_cv.layers.Resizing(
    IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True
)
train_ds = train_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

# Visualize the images in the batch
image_batch = next(iter(eval_ds.take(1)))["images"]
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)


def unpackage_dict(inputs):
    """
    Unpackages a dictionary and returns the values of the 'images' and 'labels' keys.

    Parameters:
    inputs (dict): A dictionary containing the 'images' and 'labels' keys.

    Returns:
    tuple: A tuple containing the values of the 'images' and 'labels' keys.
    """
    return inputs["images"], inputs["labels"]


# Unpackage the dictionaries in the train and test sets
train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

"""
## Define Optimizer and Loss

In this section, we define the optimizer and loss function for our model training.

We use the RMSProp optimizer with a learning rate of 1e-4. This optimizer is known for its ability to handle non-stationary objectives and adapt to changing conditions during training. The loss function we use is binary cross entropy. This loss function is commonly used for binary classification problems, where each sample belongs to one of two classes.

By defining the optimizer and loss function, we set the foundation for our model training process and ensure that the model is optimized to minimize the loss and improve its performance.
"""

# Optimizer function for the model
optimizer_fn = optimizers.RMSprop(learning_rate=lr)

# Loss function for the model
loss_fn = losses.BinaryCrossentropy(label_smoothing=0.1)

"""
## Set up Callbacks

We use callbacks to enhance the training process and monitor the model's performance. Here are the callbacks we use in this notebook:

- ModelCheckpoint: This callback saves the model weights after each epoch if the validation accuracy improves. It helps us keep track of the best model during training.

- EarlyStopping: This callback monitors the validation loss and stops the training if the loss does not improve after a certain number of epochs. It helps us prevent overfitting and saves training time.

- ReduceLROnPlateau: This callback reduces the learning rate when the validation loss plateaus. It helps us fine-tune the model and improve its performance.

- TensorBoard: This callback logs the training and validation loss to TensorBoard, which allows us to visualize and analyze the training process.

By using these callbacks, we can optimize the training process and improve the performance of our model.
"""

# Callbacks for the training process
train_callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="model.h5",
        monitor="val_accuracy",
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
    keras.callbacks.TensorBoard(log_dir="./logs"),
]

"""
## Transfer Learning from Pre-trained Weights

We use transfer learning to initialize the model weights with pre-trained weights. By using pre-trained weights, we can leverage the knowledge gained from training on a large dataset to solve our own problem. This allows us to train a model with less data and less time, while still being able to achieve good performance.
"""


def build_model(num_classes):
    """
    Build a model for malaria diagnosis using EfficientNet.

    Args:
        num_classes (int): The number of classes for classification.

    Returns:
        keras.Model: The compiled model.

    """
    # Create the base model from the pre-trained model EfficientNet
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top of pretrained model
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="pred")(x)

    # Compile the model
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = optimizer_fn
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    return model


# Model summary
model = build_model(NUM_CLASSES)
model.summary()

# Model plot
keras.utils.plot_model(
    model,
    to_file="efficientnet_model_plot.png",
    show_shapes=True,
    show_layer_names=True,
)

"""
### Train the Model
"""

# Model training
history = model.fit(
    train_ds, epochs=EPOCHS, callbacks=train_callbacks, validation_data=eval_ds
)

"""
### Plot the Training and Validation Metrics

Plot the training and validation accuracy/loss curves of the model. We can use these plots to check if the model has overfitted or underfitted the training data.
"""


def plotmodelhistory(history):
    """
    Plots the model accuracy and loss for the training and validation sets.

    Parameters:
    history (keras.callbacks.History): The history object returned by the model.fit() function.

    Returns:
    None
    """

    # Plotting the model accuracy and loss for the training and validation sets
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # summarize history for accuracy
    axs[0].plot(history.history["accuracy"])
    axs[0].plot(history.history["val_accuracy"])
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(["train", "validate"], loc="upper left")

    # summarize history for loss
    axs[1].plot(history.history["loss"])
    axs[1].plot(history.history["val_loss"])
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(["train", "validate"], loc="upper left")
    plt.show()


# list all data in history
print(history.history.keys())

plotmodelhistory(history)

"""
### Evaluate the Model
"""

# Evaluate the model on the test set
accuracy = model.evaluate(eval_ds)[1] * 100
print("Accuracy: {:.2f}%".format(accuracy))

"""
## Conclusion

- Model accuracy of EfficentNetB0 in the training set is 93.20% and in the validation set is 94.81%.
"""
