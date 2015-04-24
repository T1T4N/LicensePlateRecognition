import cv2

import os
from detector import ThresholdBlurDetector
from utils.loader import load_images, get_images_from_dir
from utils.display import get_parts_of_image, display_rectangles, show_image


def main():
    """
    Load images from a directory and process them to extract the license plate numbers
    """

    print('OpenCV version: %s' % cv2.__version__)

    # image_names = get_images_from_dir('images')
    # image_names = ['images' + os.sep + '19.jpg']
    # image_names = ['images' + os.sep + '27.jpg']
    # image_names = ['images' + os.sep + '05.jpg']
    # image_names = ['images' + os.sep + '05.jpg', 'images' + os.sep + '10.jpg', 'images' + os.sep + '11.jpg']

    image_names = ['images' + os.sep + '10.jpg']

    images = load_images(image_names)
    for src in images:
        detector = ThresholdBlurDetector(src)
        # detector = MorphologyTransformDetector(src)
        rectangles = detector.find_rectangles()
        display_rectangles(src, rectangles)

        plates = get_parts_of_image(src, rectangles)

        for image_part in plates:
            # show_image(image_part, resize=True)

            # TODO: final processing of the image before OCR
            # processed_image = process_plate_image(image_part)

            # TODO: perform OCR on the image
            # tr = TextRecognizer(processed_image)
            # tr.find_text()
            pass

if __name__ == '__main__':
    main()