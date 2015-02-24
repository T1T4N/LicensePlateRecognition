import cv2
import numpy as np

from detector import AbstractDetector
from utils import loader, display


class MorphologyTransformDetector(AbstractDetector):

    def __init__(self, image):
        if type(image) == str:
            self.image = loader.load_images(image)[0]
        elif type(image) == np.ndarray:
            self.image = image.copy()
        else:
            print("Incorrect variable type")
            # self.kernel9 = np.ones((7, 7), np.uint8)
            # self.kernel7 = np.ones((7, 7), np.uint8)
            # self.kernel5 = np.ones((7, 7), np.uint8)
            # self.
            #
            # kernel3 = np.ones((7, 7), np.uint8)

    def _check_sizes(self, candidate):
        # TODO: Filter rectangles if too big or small, or incorrect ratio
        '''
        error = 0.4
        # Macedonian car plate size: 52x11 aspect 4,72727272727
        aspect = 4.72727272727

        # Set a min and max area. All other patches are discarded
        min = 15*aspect*15
        max = 125*aspect*125

        # Get only patches that match to a respect ratio.
        rmin = aspect-aspect*error
        rmax = aspect+aspect*error

        area = candidate[0][0] * candidate[0][1]
        r = float(candidate[0][0]) / float(candidate[0][1])
        if r < 1:
            r = 1/r

        if (area < min or area > max) or (r < rmin or r > rmax):
            return False
        else:
            return True
        '''
        return True

    def find_rectangles(self):
        # create a grayscale version of the image
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        gray_img = cv2.adaptiveBilateralFilter(gray_img, (11, 11), 100)
        # gray_img = cv2.bilateralFilter(gray_img, 5, 100, 100)
        # gray_img = cv2.blur(gray_img, (5, 5))
        if __debug__:
            display.show_image(gray_img, 'Gray')

        # Apply Sobel filter on the image
        sobeled = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, ksize=3, scale=1, delta=0)
        if __debug__:
            display.show_image(sobeled, 'Sobel')

        # Do stuff here so the white is bigger
        # sobeled = cv2.morphologyEx(sobeled, cv2.MORPH_TOPHAT, kernel3)
        # show_image(sobeled)

        # Apply Otsu's Binary Thresholding
        ret, otsu_thresholded = cv2.threshold(sobeled, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        if __debug__:
            display.show_image(otsu_thresholded, 'Otsu Threshold')

        # Create a 29x3 Kernel
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 3))

        # Apply the Close morphology Transformation
        closed_otsu_thresholded = cv2.morphologyEx(otsu_thresholded, cv2.MORPH_CLOSE, element)
        if __debug__:
            display.show_image(closed_otsu_thresholded, 'Closed Morphology')

        # Find the contours in the image
        contours, hierarchy = cv2.findContours(closed_otsu_thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if __debug__:
            self.display_contours(contours)

        rects = []
        for itc in contours:
            mr = cv2.minAreaRect(itc)
            if self._check_sizes(mr):
                box = cv2.cv.BoxPoints(mr)
                rects.append(np.int0(box))
        return rects