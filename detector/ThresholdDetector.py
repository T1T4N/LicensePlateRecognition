import cv2
import numpy as np

from detector import AbstractDetector
from utils import loader, display


class ThresholdDetector(AbstractDetector):

    def __init__(self, image):
        if type(image) == str:
            self.image = loader.load_images(image)[0]
        elif type(image) == np.ndarray:
            self.image = image.copy()
        else:
            print("Incorrect variable type")

    def _check_sizes(self, candidate):
        return True

    def find_rectangles(self):
        """Finds the contours which are convex rectangles
        :returns The a list of all rectangle contours"""

        # create a grayscale version of the image
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # # Blur the image
        # gray_img = cv2.bilateralFilter(gray_img, 5, 100, 100)

        gray_img = cv2.adaptiveBilateralFilter(gray_img, (7, 7), 15) #75
        # gray_img = cv2.adaptiveBilateralFilter(gray_img, (3, 3), 100) #so dvoen blur se dobivat podobri rezultati, vekje istaknatite rabovi uste povekje se zadebeluvaat


        # gray_img = cv2.GaussianBlur(gray_img, (3,3), 3)
        if __debug__:
            display.show_image(gray_img, 'Gray')

        thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        if __debug__:
            display.show_image(thresh, 'Threshold')




        blurred = thresh

        # Find the contours in the image
        contours, hierarchy = cv2.findContours(blurred
                                               , cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if __debug__:
            self.display_contours(contours)


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

    def text_detection_mser(img_name):
        # finding text in an image with MSER
        img=cv2.imread(img_name)
        mser = cv2.MSER(_min_area=150, _max_area=2000)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()

        regions = mser.detect(gray, None)
        mserRegionsPixels = np.concatenate(regions, axis=0)

        mserMask = np.zeros((len(gray), len(gray[0]))) # ne mozi so dtype=int izleguva cela slika crna
        ind = np.add(mserRegionsPixels[:,1],(mserRegionsPixels[:,0]-1)*len(mserMask))
        for i in ind:
            length = len(mserMask)
            col = int(i/length)
            row = i - (col*length)
            col += 1
            mserMask[row, col] = 1


        # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        # cv2.polylines(gray, hulls, 1, (0, 255, 0))
        # print mserMask.shape
        edge = cv2.Canny(gray, 255, 255)

        edgeAndMSERIntersection = np.zeros((len(edge), len(edge[0])))
        for i in range(len(edge)):
            for j in range(len(edge[0])):
                if edge[i, j] >= 1 and mserMask[i, j] == 1:
                    edgeAndMSERIntersection[i, j] = 1

        #dialtion
        kernel = np.ones((1,1),np.uint8)
        closing = cv2.morphologyEx(edgeAndMSERIntersection, cv2.MORPH_OPEN, kernel)

        cv2.imshow('img', closing )
        cv2.waitKey(0)