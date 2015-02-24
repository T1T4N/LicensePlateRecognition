import os

import cv2


def load_images(filenames):
    """:returns a list of cv2::Mat objects that have been loaded from a list of filenames"""
    ret = []
    for fn in filenames:
        ret.append(cv2.imread(fn))
    return ret


def get_images_from_dir(directory):
    ret = []
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(".jpg"):
            ret.append(directory + os.sep + file_name)
    return ret