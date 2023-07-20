"""
This module contains helper functions for TensorFlow.

Author: -
Email: valverdesolera@gmail.com
Date: Jul 18, 2023
"""
import zipfile
from typing import List
from datetime import datetime
import os
import matplotlib.pyplot as plt  # type: ignore
import tensorflow  # type: ignore
from matplotlib.image import imread  # type: ignore
import requests  # type: ignore
import numpy as np


# Function to import an image and resize it to be able to be used with a model
def load_and_prep_image(filename: str, img_shape: int = 224, scale: bool = True,
                        color_channels: int = 3) -> tensorflow.Tensor:
    """
    Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, color_channels)
    :param filename: str. Filename of target image
    :param img_shape: int. Image shape
    :param scale: bool. Whether to scale pixel values to range 0-1 or not
    :param color_channels: int. Number of color channels
    :return: tensorflow.Tensor. Processed image of shape (img_shape, img_shape, color_channels)
    """
    # Reading the image
    image_file = tensorflow.io.read_file(filename)
    # Decoding the image
    decoded_image = tensorflow.image.decode_image(image_file, channels=color_channels)
    # Resizing the image
    resized_image = tensorflow.image.resize(decoded_image, [img_shape, img_shape])
    if scale:
        # Scale the image (values between 0 and 1)
        scaled_image = resized_image / 224.
        return scaled_image
    return resized_image


def predict_and_plot(model: tensorflow.keras.Model, filename: str, class_names: List[str]) -> None:
    """
    Imports an image located at filename, makes a prediction on it with a trained model and plots the image with
    the predicted class as the title.
    :param model: Model. Trained model to make predictions with (should be trained on image data)
    :param filename: str. Target image filename
    :param class_names: list. List of all class names in the dataset where the model was trained on
    :return: None
    """
    # Import the target image and preprocess it
    image = load_and_prep_image(filename)
    # Make a prediction
    prediction = model.predict(tensorflow.expand_dims(image, axis=0))
    # Get the predicted class
    if len(prediction[0]) > 1:
        predicted_class = class_names[prediction.argmax()]
    else:
        predicted_class = class_names[int(tensorflow.round(prediction)[0][0])]

    # Plot the images and predicted class
    plt.imshow(image)
    plt.title(f"Predicted class: {predicted_class}")
    plt.axis(False)


def create_tensorboard_callback(dir_name: str, experiment_name: str) -> tensorflow.keras.callbacks.TensorBoard:
    """
    Creates a TensorBoard callback instance to store log files. The log files are stored in a
    directory named "dir_name/experiment_name/current_datetime/"
    :param dir_name: str. Directory where to store log files
    :param experiment_name: str. Name of the experiment
    :return: TensorBoard. TensorBoard callback instance
    """
    log_dir = os.path.join(dir_name, experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def plot_loss_curves(history: tensorflow.keras.callbacks.History) -> None:
    """
    Plots the loss curves of a trained model for training and validation metrics
    :param history: History. TensorFlow History object
    :return: None
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(history.history["loss"]))
    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def unzip_data(filename: str) -> None:
    """
    Unzips filename into the current working directory
    :param filename: str. Path to target zip file
    :return: None
    """
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()


def walk_through_dir(dir_path: str) -> None:
    """
    Walks through dir_path and prints out the number of files and subdirectories it contains
    :param dir_path: str. Path to target directory
    :return: None
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def download_file_from_url(url: str) -> None:
    """
    Downloads a file from url
    :param url: str. URL of target file
    :return: None
    """
    file_name = os.path.basename(url)
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, "wb") as file:
            file.write(response.content)
        print(f"File {file_name} downloaded successfully.")
    else:
        print(f"Error downloading file {file_name}. Status code: {response.status_code}")


def print_random_image_from_path(path_dir: str, dataset: tensorflow.data.Dataset) -> None:
    """
    Prints a random image from path_dir
    :param path_dir: str. Path to target directory
    :param dataset: tensorflow.data.Dataset. Dataset containing the class names
    :return: None.
    """
    target_class = np.random.choice(dataset.class_names)
    target_dir = os.path.join(path_dir, target_class)
    random_image = np.random.choice(os.listdir(target_dir))
    random_image_path = os.path.join(target_dir, random_image)
    image = imread(random_image_path)
    plt.imshow(image)
    plt.axis(False)


def compare_histories(original_history: tensorflow.keras.callbacks.History,
                      new_history: tensorflow.keras.callbacks.History, initial_epochs: int = 5) -> None:
    """
    Compares two TensorFlow model History objects
    :param original_history: tensorflow.keras.callbacks.History. History object from original model (before new_history)
    :param new_history: tensorflow.keras.callbacks.History. History object from continued model training (after original_history)
    :param initial_epochs: int. Number of epochs in original_history (new_history plot starts from here)
    :return: None.
    """
    # Combine the `original_history` with the `new_history`
    total_accuracy = original_history.history["accuracy"] + new_history.history["accuracy"]
    total_loss = original_history.history["loss"] + new_history.history["loss"]
    total_val_accuracy = original_history.history["val_accuracy"] + new_history.history["val_accuracy"]
    total_val_loss = original_history.history["val_loss"] + new_history.history["val_loss"]
    # Make a plot for Accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_accuracy, label="Training Accuracy")
    plt.plot(total_val_accuracy, label="Validation Accuracy")
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    # Make a plot for Loss
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Validation Loss")
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
