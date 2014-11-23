import cv2


def load_images(filenames):
    """:returns a list of cv2::Mat objects that have been loaded from a list of filenames"""
    ret = []
    for fn in filenames:
        ret.append(cv2.imread(fn))
    return ret