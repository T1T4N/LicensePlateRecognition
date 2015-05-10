__author__ = 'robert'
import cv2

from detector import AbstractDetector
from utils import loader, display, image


class CannyDetector(AbstractDetector):
    """
    Detector that uses canny edge detection to detect license plates
    """

    def __init__(self, image, label=""):
        """
        Initialize the detector with an image an a label

        :type image: numpy.ndarray
        :param image: Image to be processed
        :type label: str
        :param label: Optional label for the image
        """

        self.image = loader.load_image(image)
        self.label = label

    def _check_size(self, candidate, area=None):
        """
        Check size with respect to aspect ratio of a standard license plate

        :type candidate: list[numpy.array]
        :param candidate: ApproxPolyDP on which to perform the check
        :rtype: bool
        :return: True if conditions satisfied, otherwise False
        """

        # TODO: Adjust error rate
        error_min = 0.17
        error_max = 0.32

        # Macedonian car plate size: 52x11 cm, aspect ratio = 4,72727272727
        aspect = float(52) / float(11)

        # TODO: Adjust coefficients for min and max area
        min_area = 15 * aspect * 15
        max_area = 112 * aspect * 112

        # Set aspect ratios with account to error.
        min_ratio = aspect - aspect * error_min
        max_ratio = aspect + aspect * error_max

        x_coordinates = [x for x in candidate[:, 0, 0]]
        y_coordinates = [y for y in candidate[:, 0, 1]]
        coordinates = [(x_coordinates[i], y_coordinates[i]) for i in range(len(x_coordinates))]

        candidate_width, candidate_height = image.calculate_size(coordinates)
        if candidate_height == 0.0 or candidate_width == 0.0:
            return False

        # Calculate candidate area
        candidate_area = candidate_height * candidate_width if area is None else area
        candidate_ratio = float(candidate_width) / float(candidate_height)
        if candidate_ratio < 1:
            candidate_ratio = 1 / candidate_ratio

        if not min_area <= candidate_area <= max_area or not min_ratio <= candidate_ratio <= max_ratio:
            print "Candidate width: %.3f, height: %.3f" % (candidate_width, candidate_height)
            print "Candidate area: %f" % candidate_area
            print "Candidate ratio: %f" % candidate_ratio
            return False
        else:
            print "Candidate width: %.3f, height: %.3f" % (candidate_width, candidate_height)
            print "Candidate area: %f" % candidate_area
            print "Candidate ratio: %f" % candidate_ratio
            print "Passed\n"
            return True

    def find_plates(self):
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

        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if __debug__:
            display.draw_contours(self.image, contours)

        rectangles = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    if self._check_size(approx):
                        rectangles.append(approx)

        processing_plates = display.get_parts_of_image(self.image, rectangles)
        ret = []

        for i, processing_plate in enumerate(processing_plates):
            processing_plate = cv2.cvtColor(processing_plate, cv2.COLOR_BGR2GRAY)
            processing_plate = cv2.bitwise_not(processing_plate)
            a, processing_plate = cv2.threshold(processing_plate, 50, 255, cv2.THRESH_OTSU)

            img_height, img_width = processing_plate.shape
            img_area = img_height * img_width

            # If the area of the plate is below 4500, perform hq2x on the plate
            if img_area < 4500:
                ret.append((
                    cv2.cvtColor(image.hq2x_zoom(processing_plate), cv2.COLOR_BGR2GRAY),
                    rectangles[i]
                ))
            else:
                ret.append((
                    processing_plate,
                    rectangles[i]
                ))

        return ret