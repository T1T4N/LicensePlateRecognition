import cv2
import numpy as np

from detector import AbstractDetector
from utils import loader, display, image


class MorphologyTransformDetector(AbstractDetector):
    """
    Detector that uses morphological image transformations to try and detect license plates
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
        candidate_area = candidate_height * candidate_width if area is None else area
        if candidate_height == 0.0 or candidate_width == 0.0:
            return False

        candidate_ratio = float(candidate_width) / float(candidate_height)
        if candidate_ratio < 1:
            candidate_ratio = 1 / candidate_ratio

        if not min_area <= candidate_area <= max_area or not min_ratio <= candidate_ratio <= max_ratio:
            return False
        else:
            if __debug__:
                print "Candidate width: %.3f, height: %.3f" % (candidate_width, candidate_height)
                print "Candidate area: %f" % candidate_area
                print "Candidate ratio: %f" % candidate_ratio
                print "Passed\n"
            return True

    def find_plates(self):
        """
        Find the license plates in the image

        :rtype: list[(numpy.array, numpy.array)]
        :return: List of tuples containing the plate image and the plate rectangle location
            The plates returned must be a grayscale image with black background and white characters
        """

        # Create a grayscale version of the image
        processing_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_height, img_width = processing_img.shape
        img_area = img_height * img_width

        # Blur the image
        processing_img = cv2.adaptiveBilateralFilter(processing_img, (11, 11), 100)

        if __debug__:
            display.show_image(processing_img, self.label, 'Gray')

        # Apply Sobel filter on the image
        sobel_img = cv2.Sobel(processing_img, cv2.CV_8U, 1, 0, ksize=3, scale=1, delta=0)
        if __debug__:
            display.show_image(sobel_img, self.label, 'Sobel')

        # sobel_img = cv2.morphologyEx(sobel_img, cv2.MORPH_TOPHAT, (3, 3))
        # if __debug__:
        # display.show_image(sobel_img)

        # Apply Otsu's Binary Thresholding
        ret, sobel_img = cv2.threshold(sobel_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        if __debug__:
            display.show_image(sobel_img, self.label, 'Otsu Threshold')

        # TODO: Variable kernel size depending on image size and/or perspective
        k_size = (50, 5)  # Kernel for a very upclose picture
        # k_size = (10, 5)  # Kernel for a distant picture
        element = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)

        # Apply the Close morphology Transformation
        sobel_img = cv2.morphologyEx(sobel_img, cv2.MORPH_CLOSE, element)
        if __debug__:
            display.show_image(sobel_img, self.label, 'Closed Morphology')

        # Find the contours in the image
        contours, hierarchy = cv2.findContours(sobel_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if __debug__:
            display.draw_contours(self.image, contours, self.label)

        rectangles = []
        for itc in contours:
            mr = cv2.minAreaRect(itc)  # Minimum enclosing rectangle
            # mr = (top-left x, top-left y), (width, height), angle-of-rotation
            box = cv2.cv.BoxPoints(mr)
            box_points = np.array([[(box[i][0], box[i][1])] for i in range(len(box))])
            if self._check_size(box_points):
                rectangles.append(np.int0(box))  # Rotated minimum enclosing rectangle

        processing_plates = display.get_parts_of_image(processing_img, rectangles)
        ret = []
        # if __debug__:
        # display.display_rectangles(self.image, rectangles)

        for i, processing_plate in enumerate(processing_plates):
            if processing_plate is not None and len(processing_plate) > 0:
                processing_plate = cv2.bitwise_not(processing_plate)
                a, processing_plate = cv2.threshold(processing_plate, 50, 255, cv2.THRESH_OTSU)

                img_width, img_height = processing_plate.shape
                img_area = img_height * img_width

                if img_area < 4500:
                    ret.append((cv2.cvtColor(image.hq2x_zoom(processing_plate), cv2.COLOR_BGR2GRAY), rectangles[i]))
                else:
                    ret.append((processing_plate, rectangles[i]))

        return ret
