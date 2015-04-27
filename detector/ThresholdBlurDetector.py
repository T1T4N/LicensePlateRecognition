import cv2
import numpy as np

from detector import AbstractDetector
from utils import loader, display, image, transform
from recognizer import TextRecognizer


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
                approx = cv2.approxPolyDP(i, 0.045 * peri, True)

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
            # deskew_line = transform.deskew_lines(processing_plate)
            deskew_text = transform.deskew_text(processing_plate)
            self._segment_contours(deskew_text)
        return rectangles

    def _segment_contours(self, plate):
        img = plate.copy()
        img2 = img.copy()
        disp_img = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)
        img_height, img_width = img.shape
        img_area = img_height * img_width

        print "\n\nSegmenting contours\nPart area: %.3f" % img_area

        boxes = []
        # RETR_CCOMP returns a two level hierarchy of the contours: parent and children
        contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i, ct in enumerate(contours):
            # mr = cv2.minAreaRect(ct)
            # box = cv2.cv.BoxPoints(mr)
            # box = np.int0(box)
            # box_points = [(box[i, 0], box[i, 1]) for i in range(len(box))]
            # box_width, box_height = image.calculate_size(box_points)

            if hierarchy[0][i][3] != -1:
                continue

            # TODO: Number one has ratio ~ 4.5: very thin and high
            x, y, box_width, box_height = cv2.boundingRect(ct)
            if box_width > 0 and box_height > 0:
                box_area = float(box_width) * float(box_height)

                box_ratio = float(box_width) / float(box_height)
                if box_ratio < 1:
                    box_ratio = 1 / float(box_ratio)

                print "Box width: %.3f, height: %.3f" % (box_width, box_height)
                print "Box area: %.3f" % box_area
                print "Box ratio: %.3f" % box_ratio
                print "Area ratio: %.3f" % (img_area / box_area)

                # TODO: Square in the middle always caught, adjust box_ratio upper limit
                if 0.5 < box_ratio < 3 and img_area / box_area < 45:
                    print "Passed\n"

                    # TODO: Fill contour without the holes
                    cv2.drawContours(img, [ct], 0, (255, 255, 255), thickness=-1)

                    # cv2.drawContours(disp_img, [box], 0, (0, 0, 255), 1)
                    cv2.rectangle(disp_img, (x, y), (x + box_width, y + box_height), (0, 255, 0), 1)

                    box_points = np.array(
                        [(x, y), (x, y + box_height), (x + box_width, y), (x + box_width, y + box_height)]
                    )
                    # box_points = sorted(box_points, key=lambda item: (item[0], item[1]))
                    boxes.append(np.array(box_points))

        # Fill holes of contours in black
        for i, ct in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                parent_idx = hierarchy[0][i][3]
                parent_contour = contours[parent_idx]
                parent_area = cv2.contourArea(parent_contour)
                child_area = cv2.contourArea(ct)
                if child_area > float(parent_area) / 9:
                    # Approximate using a polygon
                    peri = cv2.arcLength(ct, True)
                    approx = cv2.approxPolyDP(ct, 0.020 * peri, True)
                    if cv2.isContourConvex(approx):
                        cv2.drawContours(img, [approx], 0, (0, 0, 0), thickness=-1)
                        # pass

        boxes_sorted = sorted(boxes, key=lambda item: (item[0][0], item[0][1]))
        boxes_sep = display.get_parts_of_image(img, boxes_sorted)
        display.show_image(disp_img)

        labels = []
        for box in boxes_sep:
            # box_mod = image.hq2x_zoom(cv2.cvtColor(box, cv2.COLOR_GRAY2BGR))
            box_mod = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
            tr = TextRecognizer(cv2.bitwise_not(box_mod))
            text, conf = tr.find_text()
            text = text.strip()
            t2 = ""
            for i in range(len(text)):
                if ord(text[i]) in range(128):
                    t2 += text[i]
            labels.append(t2)
            print t2, conf
            # display.show_image(box_mod)

        display.multi_plot(boxes_sep, labels, 1, len(boxes_sep))
