import numpy as np
import cv2
import tesseract
import cv2.cv as cv

from detector import ThresholdBlurDetector
from utils.loader import load_images, get_images_from_dir
from utils.display import display_rectangles
from utils import display


def process_image(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.bitwise_not(img, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveBilateralFilter(gray, (7, 7), 15)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    display.show_image(thresh, 'White')


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (0,0,0)
    upper = (0,0,255)

    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask)
    # ret,gray = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    display.show_image(output, 'White')


def tesseract(img_name):
    #tries to find text in ana image
    #the image must contain only letters, else it will fail to recognize text

    image0=cv2.imread(img_name)

    #### you may need to thicken the border in order to make tesseract feel happy to ocr your image #####
    offset=20
    height,width,channel = image0.shape

    image1=cv2.copyMakeBorder(image0,offset,offset,offset,offset,cv2.BORDER_CONSTANT,value=(255,255,255))
    cv2.namedWindow("Test")
    cv2.imshow("Test", image1)
    cv2.waitKey(0)
    cv2.destroyWindow("Test")

    #####################################################################################################
    api = tesseract.TessBaseAPI()
    api.Init(".","eng",tesseract.OEM_DEFAULT)
    api.SetPageSegMode(tesseract.PSM_AUTO)
    height1,width1,channel1=image1.shape
    print image1.shape
    print image1.dtype.itemsize
    width_step = width*image1.dtype.itemsize
    print width_step

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's Binary Thresholding
    ret, image1 = cv2.threshold(image1, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # image1 = cv2.adaptiveBilateralFilter(image1, (7, 7), 15)
    # image1 = cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    # ret, image1 = cv2.threshold(image1, 110, 255, cv2.THRESH_BINARY_INV)
    # image1 = cv2.Canny(image1, 100, 100, 3)

    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 3))
    # cv2.morphologyEx(image1, cv2.MORPH_OPEN, element)

    # image1 = cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    cv2.namedWindow("Test")
    cv2.imshow("Test", image1)
    cv2.waitKey(0)
    cv2.destroyWindow("Test")


    #method 1
    iplimage = cv.CreateImageHeader((width1,height1), cv.IPL_DEPTH_8U, channel1)
    cv.SetData(iplimage, image1.tostring(),image1.dtype.itemsize * channel1 * (width1))
    tesseract.SetCvImage(iplimage,api)

    text=api.GetUTF8Text()
    conf=api.MeanTextConf()
    image=None
    print "..............."
    print "Ocred Text: %s"%text
    print "Cofidence Level: %d %%"%conf

    #method 2:
    cvmat_image=cv.fromarray(image1)
    iplimage =cv.GetImage(cvmat_image)
    print iplimage

    tesseract.SetCvImage(iplimage,api)
    #api.SetImage(m_any,width,height,channel1)
    text=api.GetUTF8Text()
    conf=api.MeanTextConf()
    image=None
    print "..............."
    print "Ocred Text: %s"%text
    print "Cofidence Level: %d %%"%conf
    api.End()


def get_parts_of_image(img, rects):
    # crops the detected recatngles in an image, and tries to find if there is text
    for rect in rects:
        xmin = 999999
        xmax = 0
        ymin = 999999
        ymax = 0
        for point in rect:
            xmax = max(xmax, point[0][0])
            ymax = max(ymax, point[0][1])
            xmin = min(xmin, point[0][0])
            ymin = min(ymin, point[0][1])
        # print ("pecati " + str(xmin) + " " + str(xmax) + " " + str(ymin) + " " + str(ymax))
        process_image(img[ymin:ymax, xmin:xmax])


def main():
    print('OpenCV version: %s' % cv2.__version__)

    image_names = get_images_from_dir('images')
    # image_names = ['images' + os.sep + '5.jpg']

    images = load_images(image_names)
    for src in images:
        # detector = MorphologyTransformDetector(src)
        # rects = detector.find_rectangles()
        # display_rectangles(src, rects)

        detector2 = ThresholdBlurDetector(src)
        rects2 = detector2.find_rectangles_martin()
        display_rectangles(src, rects2)

        get_parts_of_image(src, rects2)

        # detector2 = ThresholdDetector(src)
        # rects2 = detector2.find_rectangles()
        # display_rectangles(src, rects2)

if __name__ == '__main__':
    main()