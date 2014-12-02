import os

import cv2

from detector import MorphologyTransformDetector, ThresholdBlurDetector
from utils.loader import load_images
from utils.display import display_rectangles


def get_images_from_dir(directory):
    ret = []
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(".jpg"):
            ret.append(directory + os.sep + file_name)
    return ret


def main():
    print('Hello')
    print('OpenCV version: %s' % cv2.__version__)

    image_names = get_images_from_dir('images')
    # image_names = ['images' + os.sep + '5.jpg']

    images = load_images(image_names)
    for src in images:
        detector = MorphologyTransformDetector(src)
        rects = detector.find_rectangles()
        display_rectangles(src, rects)

        detector2 = ThresholdBlurDetector(src)
        rects2 = detector2.find_rectangles()
        display_rectangles(src, rects2)

if __name__ == '__main__':
    main()