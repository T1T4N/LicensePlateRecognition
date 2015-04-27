import cv2
import numpy as np

from detector import AbstractDetector
from utils import loader, display, image


class MorphologyTransformDetector(AbstractDetector):
    """
    Detector that uses morphological image transformations to try and detect license plates
    """

    def __init__(self, image):
        """
        Initialize detector with the given image
        :param image: String or cv2::Mat object from from which to initialize detector
        """
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

        candidate_width, candidate_height = image.calculate_size(coordinates)

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

    def find_rectangles(self):
        # self.kernel9 = np.ones((7, 7), np.uint8)
        # self.kernel7 = np.ones((7, 7), np.uint8)
        # self.kernel5 = np.ones((7, 7), np.uint8)
        # self.kernel3 = np.ones((7, 7), np.uint8)

        # create a greyscale version of the image
        processing_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        processing_img = cv2.adaptiveBilateralFilter(processing_img, (11, 11), 100)
        # grey_img = cv2.bilateralFilter(grey_img, 5, 100, 100)
        # grey_img = cv2.blur(grey_img, (5, 5))

        if __debug__:
            display.show_image(processing_img, 'Grey')

        # Apply Sobel filter on the image
        processing_img = cv2.Sobel(processing_img, cv2.CV_8U, 1, 0, ksize=3, scale=1, delta=0)
        if __debug__:
            display.show_image(processing_img, 'Sobel')

        # TODO: Try to enlarge white area
        # sobel_img = cv2.morphologyEx(sobel_img, cv2.MORPH_TOPHAT, kernel3)
        # show_image(sobel_img)

        # Apply Otsu's Binary Thresholding
        ret, processing_img = cv2.threshold(processing_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        if __debug__:
            display.show_image(processing_img, 'Otsu Threshold')

        # TODO: Variable kernel size depending on image size and/or perspective
        k_size = (29, 3)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)

        # Apply the Close morphology Transformation
        processing_img = cv2.morphologyEx(processing_img, cv2.MORPH_CLOSE, element)
        if __debug__:
            display.show_image(processing_img, 'Closed Morphology')

        # Find the contours in the image
        contours, hierarchy = cv2.findContours(processing_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if __debug__:
            display.draw_contours(self.image, contours)

        rectangles = []
        for itc in contours:
            mr = cv2.minAreaRect(itc)  # Minimum enclosing rectangle
            # itc = (top-left x, top-left y), (width, height), angle-of-rotation
            if self._check_size(mr):
                box = cv2.cv.BoxPoints(mr)
                rectangles.append(np.int0(box))  # Rotated minimum enclosing rectangle

        processing_plates = display.get_parts_of_image(processing_img, rectangles)
        ret = []

        for i, processing_plate in enumerate(processing_plates):
            img_height, img_width = processing_plate.shape
            img_area = img_height * img_width
            # TODO: do not include some parts based on different parameters
            if img_area < 4500:
                ret.append((cv2.cvtColor(image.hq2x_zoom(processing_plate), cv2.COLOR_BGR2GRAY), rectangles[i]))
            else:
                ret.append((processing_plate, rectangles[i]))

        return ret
