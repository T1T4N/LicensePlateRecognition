import cv2
import numpy as np

from detector import AbstractDetector
from utils import loader, display


class MorphologyTransformDetector(AbstractDetector):
    """
    Detector that uses morphological image transformations to try and detect license plates
    """

    def __init__(self, image):
        """
        Initialize detector with the given image
        :param image: String or cv2::Mat object from from which to initialize detector
        """

        if type(image) == str:
            self.image = loader.load_images(image)[0]
        elif type(image) == np.ndarray:
            self.image = image.copy()
        else:
            raise ValueError("Incorrect variable type")

    def _check_sizes(self, candidate):
        """
        Check size with respect to aspect ratio of a standard license plate

        :param candidate: Rectangle on which to perform the check
        :return: True if conditions satisfied, otherwise False
        """

        error = 0.4
        # Macedonian car plate size: 52x11c cm, aspect ratio = 4,72727272727
        aspect = 4.72727272727

        # Set a min and max area
        min_area = 15 * aspect * 15
        max_area = 125 * aspect * 125

        # Set aspect ratios with account to error.
        ratio_min = aspect - aspect * error
        ratio_max = aspect + aspect * error

        candidate_area = candidate[0][0] * candidate[0][1]
        candidate_ratio = float(candidate[0][0]) / float(candidate[0][1])
        if candidate_ratio < 1:
            candidate_ratio = 1 / candidate_ratio

        # TODO: Filter too big or too small rectangles or with incorrect ratio
        '''
        if (candidate_area < min_area or candidate_area > max_area)
            or (candidate_ratio < ratio_min or candidate_ratio > ratio_max):
            return False
        else:
            return True
        '''
        return True

    def find_rectangles(self):
        # self.kernel9 = np.ones((7, 7), np.uint8)
        # self.kernel7 = np.ones((7, 7), np.uint8)
        # self.kernel5 = np.ones((7, 7), np.uint8)
        # self.kernel3 = np.ones((7, 7), np.uint8)

        # create a greyscale version of the image
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        gray_img = cv2.adaptiveBilateralFilter(gray_img, (11, 11), 100)
        # gray_img = cv2.bilateralFilter(gray_img, 5, 100, 100)
        # gray_img = cv2.blur(gray_img, (5, 5))

        if __debug__:
            display.show_image(gray_img, 'Gray')

        # Apply Sobel filter on the image
        sobel_img = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, ksize=3, scale=1, delta=0)
        if __debug__:
            display.show_image(sobel_img, 'Sobel')

        # TODO: Try to enlarge white area
        # sobel_img = cv2.morphologyEx(sobel_img, cv2.MORPH_TOPHAT, kernel3)
        # show_image(sobel_img)

        # Apply Otsu's Binary Thresholding
        ret, thresholded_img = cv2.threshold(sobel_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        if __debug__:
            display.show_image(thresholded_img, 'Otsu Threshold')

        # TODO: Variable kernel size depending on image size and/or perspective
        k_size = (29, 3)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)

        # Apply the Close morphology Transformation
        closed_otsu_thresholded = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, element)
        if __debug__:
            display.show_image(closed_otsu_thresholded, 'Closed Morphology')

        # Find the contours in the image
        contours, hierarchy = cv2.findContours(closed_otsu_thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if __debug__:
            display.draw_contours(contours)

        rectangles = []
        for itc in contours:
            mr = cv2.minAreaRect(itc)  # Minimum enclosing rectangle
            if self._check_sizes(mr):
                box = cv2.cv.BoxPoints(mr)
                rectangles.append(np.int0(box))  # Rotated minimum enclosing rectangle
        return rectangles
