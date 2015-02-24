import os

import cv2


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


def get_images_from_dir(directory):
    """
    Get the names of every JPG image in a directory

    :param directory: String representing an absolute or relative path to a directory
    :return: List of strings containing names of the JPG images in the directory
    """

    ret = []
    files = os.listdir(directory)
    for file_name in files:
        if file_name.lower().endswith(".jpg"):
            ret.append(directory + os.sep + file_name)
    return ret