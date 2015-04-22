import math

import cv2

from detector import AbstractDetector
from utils import loader, display


class ThresholdBlurDetector(AbstractDetector):

    def __init__(self, image):
        self.image = loader.load_image(image)

    def _check_sizes(self, candidate):
        """
        Check size with respect to aspect ratio of a standard license plate

        :param candidate: ApproxPolyDP on which to perform the check
        :return: True if conditions satisfied, otherwise False
        """

        error = 0.4
        # Macedonian car plate size: 52x11c cm, aspect ratio = 4,72727272727
        aspect = 4.72727272727

        # Set a min and max area
        min_area = 15 * aspect * 15
        max_area = 125 * aspect * 125

        # Set aspect ratios with account to error.
        min_ratio = aspect - aspect * error
        max_ratio = aspect + aspect * error

        x_coordinates = [x for x in candidate[:, 0, 0]]
        y_coordinates = [y for y in candidate[:, 0, 1]]
        coordinates = [(x_coordinates[i], y_coordinates[i]) for i in range(len(x_coordinates))]
        coords_xsorted = sorted(coordinates, key=lambda item: (item[0], item[1]))

        candidate_width, candidate_height = self._calculate_size(coords_xsorted)

        candidate_area = candidate_height * candidate_width
        candidate_ratio = float(candidate_width) / float(candidate_height)
        if candidate_ratio < 1:
            candidate_ratio = 1 / candidate_ratio

        print "Candidate area: %f" % candidate_area
        print "Candidate ratio: %f" % candidate_ratio

        if (candidate_area < min_area or candidate_area > max_area) \
                or (candidate_ratio < min_ratio or candidate_ratio > max_ratio):
            print "Failed\n"
            return False
        else:
            print "Passed\n"
            return True

    def _calculate_size(self, coords):
        """
        Given an array of four coordinates calculates width and height of the rectangle they form

        :param coords: An array of 4 (x, y) tuples
        :return: A (width, height) tuple
        """

        h1 = math.hypot(coords[0][0] - coords[1][0], coords[0][1] - coords[1][1])
        h2 = math.hypot(coords[2][0] - coords[3][0], coords[2][1] - coords[3][1])
        height = max(h1, h2)

        d11 = abs(coords[0][1] - coords[2][1])
        d12 = abs(coords[0][1] - coords[3][1])

        d21 = abs(coords[1][1] - coords[2][1])
        d22 = abs(coords[1][1] - coords[3][1])

        width = -1
        if d11 < d12:
            width = max(width, math.hypot(coords[0][0] - coords[2][0], coords[0][1] - coords[2][1]))
        else:
            width = max(width, math.hypot(coords[0][0] - coords[3][0], coords[0][1] - coords[3][1]))

        if d21 < d22:
            width = max(width, math.hypot(coords[1][0] - coords[2][0], coords[1][1] - coords[2][1]))
        else:
            width = max(width, math.hypot(coords[1][0] - coords[3][0], coords[1][1] - coords[3][1]))

        return width, height

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
            display.draw_contours(self.image, contours)

        biggest = None
        max_area = 0
        rectangles = []

        for i in contours:
            area = cv2.contourArea(i)
            if area > 200:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.045*peri, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    if self._check_sizes(approx):
                        rectangles.append(approx)
                        if area > max_area:
                            biggest = approx
                            max_area = area

        return rectangles
