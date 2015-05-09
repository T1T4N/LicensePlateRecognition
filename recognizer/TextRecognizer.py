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

        :rtype: (str, int)
        :return: Detected text with confidence level
        """

        image0 = self.image

        # Thicken the border in order to make Tesseract feel happy to OCR the image
        offset = 20
        height, width, channel = image0.shape

        image1 = cv2.copyMakeBorder(image0, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        api = tesseract.TessBaseAPI()
        api.Init(".", "eng", tesseract.OEM_DEFAULT)
        # Allow only the specified characters
        api.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        # Assume a single uniform block of text
        api.SetPageSegMode(tesseract.PSM_SINGLE_BLOCK)

        height1, width1, channel1 = image1.shape
        width_step = width * image1.dtype.itemsize

        # Method 1
        iplimage = cv2.cv.CreateImageHeader((width1, height1), cv2.cv.IPL_DEPTH_8U, channel1)
        cv2.cv.SetData(iplimage, image1.tostring(), image1.dtype.itemsize * channel1 * width1)
        tesseract.SetCvImage(iplimage, api)
        text = api.GetUTF8Text()
        conf = api.MeanTextConf()

        # Method 2:
        cvmat_image = cv2.cv.fromarray(image1)
        iplimage = cv2.cv.GetImage(cvmat_image)
        tesseract.SetCvImage(iplimage, api)
        text2 = api.GetUTF8Text()
        conf2 = api.MeanTextConf()

        api.End()
        if text == text2:
            return text, conf
        else:
            return (text, conf), (text2, conf2)
