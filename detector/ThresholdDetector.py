import cv2
import numpy as np

from detector import AbstractDetector
from utils import loader, display


class ThresholdDetector(AbstractDetector):

    def __init__(self, image):
        self.image = loader.load_image(image)

    def _check_sizes(self, candidate):
        return True

    def find_rectangles(self):
        """
        Find the contours which are convex rectangles

        :return: List of all found rectangle contours
        """

        # create a greyscale version of the image
        grey_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        grey_img = cv2.adaptiveBilateralFilter(grey_img, (7, 7), 15)  # 75

        # So dvoen blur se dobivat podobri rezultati, vekje istaknatite rabovi uste povekje se zadebeluvaat
        # grey_img = cv2.adaptiveBilateralFilter(grey_img, (3, 3), 100)

        # grey_img = cv2.GaussianBlur(grey_img, (3,3), 3)

        if __debug__:
            display.show_image(grey_img, 'Gray')

        thresh_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        if __debug__:
            display.show_image(thresh_img, 'Threshold')

        blurred_img = thresh_img

        # Find the contours in the image
        contours, hierarchy = cv2.findContours(blurred_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if __debug__:
            display.draw_contours(contours)

        biggest = None
        max_area = 0
        rectangles = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > 200:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.045*peri, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    rectangles.append(approx)
                    if area > max_area:
                        biggest = approx
                        max_area = area

        return rectangles

    def text_detection_mser(self, img_name):
        """
        Find text in an image with MSER

        :param img_name: String containing the name of the image
        :return:
        """

        img = cv2.imread(img_name)
        mser = cv2.MSER(_min_area=150, _max_area=2000)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()

        regions = mser.detect(gray, None)
        mser_regions_pixels = np.concatenate(regions, axis=0)
        mser_mask = np.zeros((len(gray), len(gray[0])))  # Ne mozi so dtype=int izleguva cela slika crna

        ind = np.add(mser_regions_pixels[:, 1], (mser_regions_pixels[:, 0] - 1) * len(mser_mask))
        for i in ind:
            length = len(mser_mask)
            col = int(i/length)
            row = i - (col*length)
            col += 1
            mser_mask[row, col] = 1

        # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        # cv2.polylines(gray, hulls, 1, (0, 255, 0))
        # print mser_mask.shape
        edge = cv2.Canny(gray, 255, 255)

        edge_and_mser_intersection = np.zeros((len(edge), len(edge[0])))
        for i in range(len(edge)):
            for j in range(len(edge[0])):
                if edge[i, j] >= 1 and mser_mask[i, j] == 1:
                    edge_and_mser_intersection[i, j] = 1

        # Dilation
        kernel = np.ones((1, 1), np.uint8)
        closing = cv2.morphologyEx(edge_and_mser_intersection, cv2.MORPH_OPEN, kernel)

        cv2.imshow('img', closing)
        cv2.waitKey(0)