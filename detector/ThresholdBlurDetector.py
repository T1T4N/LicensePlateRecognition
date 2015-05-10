import cv2

from detector import AbstractDetector
from utils import loader, display, image


class ThresholdBlurDetector(AbstractDetector):
    """
    Detector that uses blurring, thresholding image transformations and finding contours to detect license plates
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
            return False
        else:
            print "Candidate width: %.3f, height: %.3f" % (candidate_width, candidate_height)
            print "Candidate area: %f" % candidate_area
            print "Candidate ratio: %f" % candidate_ratio
            print "Passed\n"
            return True

    def _filter_white(self, processing_plate, mask_pixels):
        """
        Filter every color pixel from a plate
        """

        white_img = mask_pixels
        # display.show_image(white_img)
        processing_copy = processing_plate.copy()
        for ii in range(len(processing_plate)):
            for jj in range(len(processing_plate[ii])):
                if white_img[ii, jj] == 255:  # if white in the filter set it to black
                    processing_copy[ii, jj] = 0
        # display.show_image(processing_copy, "processed_white")
        return processing_copy

    def find_plates(self):
        """
        Find the license plates in the image

        :rtype: list[(numpy.array, numpy.array)]
        :return: List of tuples containing the plate image and the plate rectangle location.
            The plates returned must be a grayscale image with black background and white characters
        """

        # Create a grayscale version of the image
        processing_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        kernel_size = (7, 7)
        processing_img = cv2.adaptiveBilateralFilter(processing_img, kernel_size, 15)
        # processing_img = cv2.GaussianBlur(processing_img, (7, 7), 3)

        if __debug__:
            display.show_image(processing_img, self.label, 'Gray')

        # Threshold the image using an adaptive algorithm
        processing_img = cv2.adaptiveThreshold(processing_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 11, 2)

        if __debug__:
            display.show_image(processing_img, self.label, 'Threshold')

        # Find the contours in the image. MODIFIES source image, hence a copy is used
        contours, hierarchy = cv2.findContours(processing_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if __debug__:
            display.draw_contours(self.image, contours, self.label)

        rectangles = []
        for i in contours:
            area = cv2.contourArea(i)  # Calculate the area of the contour
            if area > 200:  # Trivial check
                peri = cv2.arcLength(i, True)  # Calculate a contour perimeter
                approx = cv2.approxPolyDP(i, 0.045 * peri, True)  # Approximate the curve using a polygon

                # Consider the polygon only if it is convex and has 4 edges
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    if self._check_size(approx):
                        rectangles.append(approx)

        processing_plates = display.get_parts_of_image(processing_img, rectangles)
        ret = []

        # Experimental: Mask every color pixel in every plate rectangle from the original picture
        # mask_pixels = display.get_white_pixels(self.image, rectangles)

        for i, processing_plate in enumerate(processing_plates):
            img_height, img_width = processing_plate.shape
            img_area = img_height * img_width

            # masked_plate = self._filter_white(processing_plate, mask_pixels[i])
            # processing_plate = masked_plate

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
