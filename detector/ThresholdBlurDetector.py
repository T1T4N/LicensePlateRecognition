import cv2
import numpy as np

from detector import AbstractDetector
from utils import loader, display


class ThresholdBlurDetector(AbstractDetector):

    def __init__(self, image):
        if type(image) == str:
            self.image = loader.load_images(image)[0]
        elif type(image) == np.ndarray:
            self.image = image.copy()
        else:
            print("Incorrect variable type")

    def _check_sizes(self, candidate):
        return True


    def find_rectangles_martin(self):
        """Finds the contours which are convex rectangles
        :returns The a list of all rectangle contours"""

        # create a grayscale version of the image
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # # Blur the image
        # gray_img = cv2.bilateralFilter(gray_img, 5, 100, 100)

        gray_img = cv2.adaptiveBilateralFilter(gray_img, (7, 7), 15) #75
        # gray_img = cv2.adaptiveBilateralFilter(gray_img, (3, 3), 100) #so dvoen blur se dobivat podobri rezultati, vekje istaknatite rabovi uste povekje se zadebeluvaat


        # gray_img = cv2.GaussianBlur(gray_img, (3,3), 3)
        if __debug__:
            display.show_image(gray_img, 'Gray')

        thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        if __debug__:
            display.show_image(thresh, 'Threshold')




        blurred = thresh

        # Find the contours in the image
        contours, hierarchy = cv2.findContours(blurred
                                               , cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if __debug__:
            self.display_contours(contours)


        biggest = None
        max_area = 0
        rectangles = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > 200:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.045*peri, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    rectangles.append(approx)
                    if area > max_area:
                        biggest = approx
                        max_area = area

        return rectangles


    def find_rectangles(self):
        """Finds the contours which are convex rectangles
        :returns The a list of all rectangle contours"""

        # create a grayscale version of the image
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        gray_img = cv2.adaptiveBilateralFilter(gray_img, (11, 11), 100)
        # gray_img = cv2.bilateralFilter(gray_img, 5, 100, 100)
        # gray_img = cv2.blur(gray_img, (5, 5))
        if __debug__:
            display.show_image(gray_img, 'Gray')

        thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)

        # gaussian 11, (3,3), 0: 4 tablicki od 6 sliki
        # gaussian 21, (3,3), 0: 8 tablicki od 13 sliki
        # gaussian 17, (3,3), 0: 10 tablicki od 13 sliki

        blurred = cv2.GaussianBlur(thresh, (3, 3), 0)

        if __debug__:
            display.show_image(blurred, 'Blurred')

        edges = cv2.Canny(blurred, 100, 100, 3)
        if __debug__:
            display.show_image(edges, 'Canny edges')

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if __debug__:
            self.display_contours(contours)

        biggest = None
        max_area = 0
        rectangles = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02*peri, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    rectangles.append(approx)
                    if area > max_area:
                        biggest = approx
                        max_area = area

        return rectangles