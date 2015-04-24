import math

import cv2
import numpy as np

from PIL import Image
from detector import AbstractDetector
from utils import loader, display, hq2x


class ThresholdBlurDetector(AbstractDetector):

    def __init__(self, image):
        self.image = loader.load_image(image)

    def _check_size(self, candidate, area=-1):
        """
        Check size with respect to aspect ratio of a standard license plate

        :param candidate: ApproxPolyDP on which to perform the check
        :return: True if conditions satisfied, otherwise False
        """

        # TODO: Adjust error rate
        error_min = 0.17
        error_max = 0.32
        # Macedonian car plate size: 52x11c cm, aspect ratio = 4,72727272727
        aspect = float(52) / float(11)

        # Set a min and max area
        # TODO: Adjust coefficients for min and max area
        min_area = 17 * aspect * 17
        max_area = 112 * aspect * 112

        # Set aspect ratios with account to error.
        min_ratio = aspect - aspect * error_min
        max_ratio = aspect + aspect * error_max

        x_coordinates = [x for x in candidate[:, 0, 0]]
        y_coordinates = [y for y in candidate[:, 0, 1]]
        coordinates = [(x_coordinates[i], y_coordinates[i]) for i in range(len(x_coordinates))]
        coords_xsorted = sorted(coordinates, key=lambda item: (item[0], item[1]))

        candidate_width, candidate_height = self._calculate_size(coords_xsorted)

        candidate_area = candidate_height * candidate_width
        candidate_ratio = float(candidate_width) / float(candidate_height)
        if candidate_ratio < 1:
            candidate_ratio = 1 / candidate_ratio

        if (candidate_area < min_area or candidate_area > max_area) \
                or (candidate_ratio < min_ratio or candidate_ratio > max_ratio):
            # print "Candidate width: %.3f, height: %.3f" % (candidate_width, candidate_height)
            # print "Candidate area: %f" % candidate_area
            # print "Candidate ratio: %f\n" % candidate_ratio
            return False
        else:
            print "Candidate width: %.3f, height: %.3f" % (candidate_width, candidate_height)
            print "Candidate area: %f" % candidate_area
            print "Candidate ratio: %f" % candidate_ratio
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

    def _deskew_lines(self, plate):
        angle_rad = 0.0
        img = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)

        # hq2x algorithm
        source = Image.fromarray(plate)
        dest = hq2x.hq2x(source)
        img2x = np.array(dest)

        display.show_image(img, resize=True)
        display.show_image(img2x, resize=True)

        img = cv2.cvtColor(img2x, cv2.COLOR_BGR2GRAY)
        disp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        height, width = img.shape

        lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=3 * width / 4, maxLineGap=20)
        if lines is not None and len(lines) > 0:
            for i in range(len(lines[0])):
                line = lines[0, i]
                x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                cv2.line(disp_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
                line_angle = math.atan2(y2 - y1, x2 - x1)
                angle_rad += line_angle
                print "Line angle: %.3f" % line_angle

            angle_rad /= len(lines[0])
            print "Avg angle rad: %.3f" % angle_rad
            angle = math.degrees(angle_rad)
            print "Avg angle deg: %.3f\n" % angle

            display.show_image(disp_img, resize=True)
            rotation_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            disp_img = cv2.warpAffine(disp_img, rotation_mat, (width, height))
            display.show_image(disp_img, resize=True)

    def find_rectangles(self):
        """
        Find the contours which are convex rectangles

        :return: List of all found rectangle contours
        """

        # Create a greyscale version of the image
        processing_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        # processing_img = cv2.bilateralFilter(processing_img, 5, 100, 100)
        processing_img = cv2.adaptiveBilateralFilter(processing_img, (7, 7), 15)  # 75

        # So dvoen blur se dobivat podobri rezultati, vekje istaknatite rabovi uste povekje se zadebeluvaat
        # processing_img = cv2.adaptiveBilateralFilter(processing_img, (3, 3), 100)
        # processing_img = cv2.GaussianBlur(processing_img, (3,3), 3)

        if __debug__:
            display.show_image(processing_img, 'Grey')

        processing_img = cv2.adaptiveThreshold(processing_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 11, 2)

        if __debug__:
            display.show_image(processing_img, 'Threshold')

        # Find the contours in the image. MODIFIES source image
        processing_copy = processing_img.copy()
        contours, hierarchy = cv2.findContours(processing_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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
                    if self._check_size(approx):
                        rectangles.append(approx)
                        if area > max_area:
                            biggest = approx
                            max_area = area

        processing_plates = display.get_parts_of_image(processing_img, rectangles)
        # TODO: do not include some parts based on different parameters
        for processing_plate in processing_plates:
            # Skew correction using lines detection
            self._deskew_lines(processing_plate)

        return rectangles
