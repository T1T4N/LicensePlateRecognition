import cv2

from detector import AbstractDetector
from utils import loader, display


class ThresholdBlurDetector(AbstractDetector):

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
        # grey_img = cv2.bilateralFilter(grey_img, 5, 100, 100)
        grey_img = cv2.adaptiveBilateralFilter(grey_img, (7, 7), 15)  # 75

        # So dvoen blur se dobivat podobri rezultati, vekje istaknatite rabovi uste povekje se zadebeluvaat
        # grey_img = cv2.adaptiveBilateralFilter(grey_img, (3, 3), 100)
        # grey_img = cv2.GaussianBlur(grey_img, (3,3), 3)

        if __debug__:
            display.show_image(grey_img, 'Grey')

        thresh_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        if __debug__:
            display.show_image(thresh_img, 'Threshold')

        blurred_img = thresh_img

        # Find the contours in the image
        contours, hierarchy = cv2.findContours(blurred_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if __debug__:
            display.draw_contours(contours)

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
