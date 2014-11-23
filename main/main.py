import cv2

from detector import MorphologyTransformDetector, ThresholdBlurDetector
from utils.loader import load_images
from utils.display import display_rectangles


def main():
    print('Hello')
    print('OpenCV version: %s' % cv2.__version__)

    #image_names = [r'images\1.JPG', r'images\2.JPG', r'images\3.JPG', r'images\4.JPG', r'images\5.JPG', r'images\6.JPG']

    # Linux version
    #image_names = [r'images/1.JPG', r'images/2.JPG', r'images/3.JPG', r'images/4.JPG', r'images/5.JPG', r'images/6.JPG', r'images/7.JPG', r'images/8.JPG', r'images/9.JPG', r'images/10.JPG', r'images/11.JPG', r'images/12.JPG', r'images/13.JPG']

    #image_names = [r'images\5.JPG']

    # Linux version
    image_names = [r'images/4.JPG']

    #kernel9 = np.ones((7, 7), np.uint8)
    #kernel7 = np.ones((7, 7), np.uint8)
    #kernel5 = np.ones((7, 7), np.uint8)
    #kernel3 = np.ones((7, 7), np.uint8)

    images = load_images(image_names)
    for src in images:
        detector = MorphologyTransformDetector(src)
        rects = detector.find_rectangles()
        display_rectangles(src, rects)

        detector2 = ThresholdBlurDetector(src)
        rects2 = detector2.find_rectangles()
        display_rectangles(src, rects2)

if __name__ == '__main__':
    main()