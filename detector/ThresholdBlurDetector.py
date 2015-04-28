import cv2

from detector import AbstractDetector
from utils import loader, display, image


class ThresholdBlurDetector(AbstractDetector):

    def __init__(self, image, label=""):
        self.image = loader.load_image(image)
        self.label = label

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
        if candidate_height == 0.0 or candidate_width == 0.0:
            return False

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
        # processing_img = cv2.GaussianBlur(processing_img, (7, 7), 3)

        if __debug__:
            display.show_image(processing_img, self.label, 'Grey')

        processing_img = cv2.adaptiveThreshold(processing_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 11, 2)

        if __debug__:
            display.show_image(processing_img, self.label, 'Threshold')

        # Find the contours in the image. MODIFIES source image
        processing_copy = processing_img.copy()
        contours, hierarchy = cv2.findContours(processing_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if __debug__:
            display.draw_contours(self.image, contours, self.label)

        rectangles = []

        for i in contours:
            area = cv2.contourArea(i)
            if area > 200:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.045 * peri, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    if self._check_size(approx):
                        rectangles.append(approx)

        processing_plates = display.get_parts_of_image(processing_img, rectangles)
        ret = []

        # getting non-gray, non-white and non-black pixels
        white_pixels = display.get_white_pixels(self.image, rectangles)

        for i, processing_plate in enumerate(processing_plates):
            img_height, img_width = processing_plate.shape
            img_area = img_height * img_width

            # filtering by color
            white_img = white_pixels[i]
            # display.show_image(white_img)
            processing_copy = processing_plate.copy()
            for ii in range(len(processing_plate)):
                for jj in range(len(processing_plate[ii])):
                    if white_img[ii, jj] == 255:  # if white in the filter set it to black
                        processing_copy[ii, jj] = 0
            # display.show_image(processing_copy, "processed_white")
            # processing_plate = processing_copy.copy()

            # TODO: do not include some parts based on different parameters
            if img_area < 4500:
                ret.append((cv2.cvtColor(image.hq2x_zoom(processing_plate), cv2.COLOR_BGR2GRAY), rectangles[i]))
            else:
                ret.append((processing_plate, rectangles[i]))

        return ret