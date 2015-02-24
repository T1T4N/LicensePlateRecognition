__author__ = 'robert'
import cv2

from detector import AbstractDetector
from utils import loader, display


class OldThresholdBlurDetector(AbstractDetector):
    def __init__(self, image):
        self.image = loader.load_image(image)

    def _check_sizes(self, candidate):
        return True

    def find_rectangles(self):
        """
        Find the contours which are convex rectangles

        :return: List of all found rectangle contours
        """

        # Create a greyscale version of the image
        grey_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        grey_img = cv2.adaptiveBilateralFilter(grey_img, (11, 11), 100)

        if __debug__:
            display.show_image(grey_img, 'Gray')

        # blockSize=11, (3,3), 0: 4 plates out of 6 pics
        # blockSize=21, (3,3), 0: 8 plates out of 13 pics
        # blockSize=17, (3,3), 0: 10 plates out of 13 pics
        blur_kernel_size = (3, 3)
        thresh = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)
        blurred = cv2.GaussianBlur(thresh, blur_kernel_size, 0)

        if __debug__:
            display.show_image(blurred, 'Blurred')

        edges = cv2.Canny(blurred, 100, 100, 3)
        if __debug__:
            display.show_image(edges, 'Canny edges')

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if __debug__:
            display.draw_contours(self.image, contours)

        rectangles = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    rectangles.append(approx)

        return rectangles