import cv2

from detector import ThresholdBlurDetector, MorphologyTransformDetector, CannyDetector
from utils.loader import load_images, get_images_from_dir
from utils.display import display_rectangles, show_image, multi_plot
from utils.transform import deskew_lines, deskew_text
from utils.segment import segment_contours
from recognizer import TextRecognizer


def main():
    """
    Load images from a directory and process them to extract the license plate numbers
    """

    print('OpenCV version: %s' % cv2.__version__)

    image_names = sorted(get_images_from_dir('images'))
    images = load_images(image_names)

    for i, src in enumerate(images):
        detector = ThresholdBlurDetector(src, image_names[i])
        # detector = CannyDetector(src, image_names[i])
        # detector = MorphologyTransformDetector(src, image_names[i])

        plates = detector.find_plates()
        display_rectangles(src, [plates[i][1] for i in range(len(plates))])

        for plate, original_rectangle in plates:
            show_image(plate, resize=True)

            # CautionL The following methods require the plate to have black background and white characters

            # Skew correction using lines detection
            # img = transform.deskew_lines(plate)

            # Skew correction using contours
            img = deskew_text(plate)

            # Cut the picture letter by letter
            boxes = segment_contours(img)

            ########################################
            # Any detected character (box) modification should be done here
            ########################################

            labels = []
            for box in boxes:
                # Inverts the character image so it has a white background and black character
                box = cv2.bitwise_not(box)

                # Initialize Tesseract for this image
                tr = TextRecognizer(box)
                text, conf = tr.find_text()  # Detect text with confidence level
                text = text.strip()

                # Cleaning the text of invalid values
                t2 = ""
                for idx in range(len(text)):
                    if ord(text[idx]) in range(128):
                        t2 += text[idx]
                # Add a label to the list
                labels.append(t2 + ", " + str(conf))

            # Display each box with a label above it
            multi_plot(boxes, labels, 1, len(boxes))

if __name__ == '__main__':
    main()