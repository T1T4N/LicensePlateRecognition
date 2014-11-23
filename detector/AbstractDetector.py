import cv2
from abc import ABCMeta
from abc import abstractmethod
from utils import display


class AbstractDetector(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def find_rectangles(self):
        pass

    @abstractmethod
    def _check_sizes(self, candidate):
        pass

    def display_contours(self, contours):
        src = self.image.copy()
        cv2.drawContours(src, contours, -1, (0, 255, 0), 1)
        display.show_image(src, 'Contours')