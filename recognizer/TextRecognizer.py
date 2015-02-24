__author__ = 'robert'
import cv2
import tesseract

from utils import loader


class TextRecognizer(object):
    """
    Utilizes the Tesseract engine to perform OCR on an image
    """

    def __init__(self, image):
        self.image = loader.load_image(image)

    def find_text(self):
        """
        Find text in an image
        The image must contain only letters, else it will fail to recognize text

        :return: List of strings containing the detected text
        """

        image0 = self.image

        # Thicken the border in order to make Tesseract feel happy to OCR the image
        offset = 20
        height, width, channel = image0.shape

        image1 = cv2.copyMakeBorder(image0, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.namedWindow("Test")
        cv2.imshow("Test", image1)
        cv2.waitKey(0)
        cv2.destroyWindow("Test")

        #####################################################################################################
        api = tesseract.TessBaseAPI()
        api.Init(".", "eng", tesseract.OEM_DEFAULT)
        api.SetPageSegMode(tesseract.PSM_AUTO)
        height1, width1, channel1 = image1.shape
        width_step = width * image1.dtype.itemsize

        print image1.shape
        print image1.dtype.itemsize
        print width_step

        # Method 1
        iplimage = cv2.cv.CreateImageHeader((width1, height1), cv2.cv.IPL_DEPTH_8U, channel1)
        cv2.cv.SetData(iplimage, image1.tostring(), image1.dtype.itemsize * channel1 * width1)
        tesseract.SetCvImage(iplimage, api)

        text = api.GetUTF8Text()
        conf = api.MeanTextConf()
        image = None

        print "..............."
        print "Ocred Text: %s" % text
        print "Cofidence Level: %d %%" % conf

        # Method 2:
        cvmat_image = cv2.cv.fromarray(image1)
        iplimage = cv2.cv.GetImage(cvmat_image)
        print iplimage

        tesseract.SetCvImage(iplimage, api)
        # api.SetImage(m_any,width,height,channel1)
        text = api.GetUTF8Text()
        conf = api.MeanTextConf()
        image = None
        print "..............."
        print "Ocred Text: %s" % text
        print "Cofidence Level: %d %%" % conf
        api.End()