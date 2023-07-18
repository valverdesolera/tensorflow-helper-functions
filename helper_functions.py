"""
This module contains helper functions for TensorFlow.

Author: -
Email: valverdesolera@gmail.com
Date: Jul 18, 2023
"""
import matplotlib.pyplot as plt
import zipfile
from typing import List
from os.path import join
from os import walk
from datetime import datetime
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg, resize
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow import expand_dims
from tensorflow import round
from tensorflow.keras.callbacks import TensorBoard, History


# Function to import an image and resize it to be able to be used with a model
def load_and_prep_image(filename: str, img_shape: int = 224, scale: bool = True,
                        color_channels: int = 3) -> tensorflow.Tensor:
    """
    Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, color_channels
    :param filename: str. Filename of target image
    :param img_shape: int. Image shape
    :param scale: bool. Whether to scale pixel values to range 0-1 or not
    :return: tensorflow.Tensor. Processed image
    """
    # Reading the image
    image_file = read_file(filename)
    # Decoding the image
    decoded_image = decode_jpeg(image_file, channels=color_channels)
    # Resizing the image
    resized_image = resize(decoded_image, [img_shape, img_shape])
    if scale:
        # Scale the image (values between 0 and 1)
        scaled_image = resized_image / 224.
        return scaled_image
    return resized_image


def predict_and_plot(model: Model, filename: str, class_names: List[str]) -> None:
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
    prediction = model.predict(expand_dims(image, axis=0))
    # Get the predicted class
    if len(prediction[0]) > 1:
        predicted_class = class_names[prediction.argmax()]
    else:
        predicted_class = class_names[int(round(prediction)[0][0])]

    # Plot the images and predicted class
    plt.imshow(image)
    plt.title(f"Predicted class: {predicted_class}")
    plt.axis(False)


def create_tensorboard_callback(dir_name: str, experiment_name: str) -> TensorBoard:
    """
    Creates a TensorBoard callback instance to store log files. The log files are stored in a
    directory named "dir_name/experiment_name/current_datetime/"
    :param dir_name: str. Directory where to store log files
    :param experiment_name: str. Name of the experiment
    :return: TensorBoard. TensorBoard callback instance
    """
    log_dir = join(dir_name, experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def plot_loss_curves(history: History) -> None:
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
    for dirpath, dirnames, filenames in walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
