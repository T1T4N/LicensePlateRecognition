import os
import cv2
import numpy as np


def get_images_from_dir(directory):
    """
    Get the names of every JPG image in a directory

    :param directory: String representing an absolute or relative path to a directory
    :return: List of strings containing names of the JPG images in the directory
    """

    ret = []
    files = os.listdir(directory)
    for file_name in files:
        if file_name.lower().endswith(".jpg") and \
                not file_name.lower().startswith("."):  # Do not consider UNIX hidden files
            ret.append(directory + os.sep + file_name)
    return ret


def load_images(filenames):
    """
    Load images represented by an array of filenames

    :param filenames: List of strings holding filenames of the images
    :return: List of cv2::Mat objects representing the images
    """

    ret = []
    for fn in filenames:
        ret.append(cv2.imread(fn))
    return ret


def load_image(image):
    """
    Loads or copies an image depending on the parameter
    :param image: String or cv2::Mat object from which to create an image
    :return: cv2::Mat object representing the picture
    """

    if type(image) == str:
        return load_images(image)[0]
    elif type(image) == np.ndarray:
        return image.copy()
    else:
        raise ValueError("Incorrect variable type")
