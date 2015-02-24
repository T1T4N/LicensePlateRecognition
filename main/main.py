import cv2

from detector import ThresholdBlurDetector
from utils.loader import load_images, get_images_from_dir
from utils.display import get_parts_of_image, display_rectangles


def main():
    print('OpenCV version: %s' % cv2.__version__)

    image_names = get_images_from_dir('images')
    # image_names = ['images' + os.sep + '5.jpg']

    images = load_images(image_names)
    for src in images:
        detector = ThresholdBlurDetector(src)
        rectangles = detector.find_rectangles()
        display_rectangles(src, rectangles)

        plates = get_parts_of_image(src, rectangles)

        for image_part in plates:
            # TODO: final processing of the image before OCR
            # processed_image = process_plate_image(image_part)

            # TODO: perform OCR on the image
            # tr = TextRecognizer(processed_image)
            # tr.find_text()
            pass

if __name__ == '__main__':
    main()