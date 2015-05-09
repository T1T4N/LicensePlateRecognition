import os
import cv2
import numpy as np


def get_images_from_dir(directory):
    """
    Get the names of every JPG image in a directory

    :type directory: str
    :param directory: Absolute or relative path to a directory
    :rtype: list[str]
    :return: Names of the JPG images in the directory
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

    :type filenames: list[str]
    :param filenames: Filenames of the images
    :rtype: list[numpy.array]
    :return: The loaded images
    """

    ret = []
    for fn in filenames:
        ret.append(cv2.imread(fn))
    return ret


def load_image(image):
    """
    Loads or copies an image depending on the parameter

    :type image: str | numpy.array
    :param image: Object from which to create an image
    :rtype: numpy.array
    :return: The loaded picture
    :raises: ValueError if image type is not str or numpy.array
    """

    if type(image) == str:
        return load_images(image)[0]
    elif type(image) == np.ndarray:
        return image.copy()
    else:
        raise ValueError("Incorrect variable type")
