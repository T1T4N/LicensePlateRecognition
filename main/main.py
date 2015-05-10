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
        detectors = [ThresholdBlurDetector(src, image_names[i]),
                     CannyDetector(src, image_names[i]),
                     MorphologyTransformDetector(src, image_names[i])
        ]

        plates_text = set([])
        for detector in detectors:
            plates = detector.find_plates()
            if __debug__:
                display_rectangles(src, [plates[i][1] for i in range(len(plates))])

            for plate, original_rectangle in plates:
                if __debug__:
                    show_image(plate, resize=True)

                # CAUTION: The following methods require the plate to have black background and white characters

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
                plate_text = ""
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
                    plate_text += t2

                # Add the detected plate text to the set for the current image
                if plate_text.strip() != "":
                    plates_text.add(plate_text.strip())

                # Display each box with a label above it
                if __debug__:
                    multi_plot(boxes, labels, 1, len(boxes))

        print "Detected plates in this picture:"
        for detected_text in plates_text:
            print detected_text
        inp = raw_input("Press any key to continue to the next picture")

if __name__ == '__main__':
    main()