import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(image, image_label="", image_title='image', resize=True):
    """
    Show the cv2::Mat image in a window and waits for a key

    :param image: Image to be shown
    :param image_label: Option image filename
    :param image_title: Optional title for the window
    """

    title = image_title + " " + image_label
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    if resize:
        cv2.resizeWindow(title, 1050, 615)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
    # cv2.imwrite('test.jpg', src)


def multi_plot(images, titles, rows, cols):
    """
    Plot the specified images with the specified titles in a matrix of images

    :param titles: Array of string representing titles for each image
    :param images: Array of images to display
    :param rows: Row count of the matrix
    :param cols: Column count of the matrix
    """

    if len(titles) != len(images):
        raise ValueError("The number of titles is different from the number of images")

    for i in xrange(len(titles)):
        # Place the image in the corresponding cell
        plt.subplot(rows, cols, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()  # Show the matrix of images


def display_rectangles(image, rectangles):
    """
    Draw convex rectangle contours with green color on top of the given image

    :param image: Image on top of which to paint the contours
    :param rectangles: Array of convex rectangles to be painted
    """

    src2 = image.copy()
    color = (0, 255, 0)
    cv2.drawContours(src2, rectangles, -1, color, 2)
    show_image(src2)


def draw_contours(image, contours, image_label=""):
    """
    Plot specified contours on the specified image
    :param image: cv2::Mat image on which to plot
    :param contours: List of contours to be plotted
    """

    src = image.copy()
    color = (0, 255, 0)
    cv2.drawContours(src, contours, -1, color, 1)
    show_image(src, image_label, 'Contours')


def get_parts_of_image(img, rectangles, points_sorted=False):
    """
    Crops the detected rectangles from the image and tries to find any text

    :param img: Source image from which to crop
    :param rectangles: List of rectangles representing crop areas
    :return: List of images representing the cropped areas
    """

    ret = []
    if not points_sorted:
        # TODO: sort points by x axis
        pass

    for rect in rectangles:
        x_min = 999999
        x_max = 0
        y_min = 999999
        y_max = 0
        for point in rect:
            if len(point.shape) == 1:
                x_max = max(x_max, point[0])
                y_max = max(y_max, point[1])
                x_min = min(x_min, point[0])
                y_min = min(y_min, point[1])
            else:
                x_max = max(x_max, point[0][0])
                y_max = max(y_max, point[0][1])
                x_min = min(x_min, point[0][0])
                y_min = min(y_min, point[0][1])
        image_part = img[y_min:y_max, x_min:x_max]
        ret.append(image_part)
    return ret


def get_white_pixels(img, rectangles, points_sorted=False):
    """
    Crops the detected rectangles from the image and tries to find any text
    used for finding which pixels are not of interest - not black, white or gray

    :param img: Source image from which to crop
    :param rectangles: List of rectangles representing crop areas
    :return: List of images representing the cropped areas
    """

    ret = []
    if not points_sorted:
        pass

    for rect in rectangles:
        x_min = 999999
        x_max = 0
        y_min = 999999
        y_max = 0
        for point in rect:
            if len(point.shape) == 1:
                x_max = max(x_max, point[0])
                y_max = max(y_max, point[1])
                x_min = min(x_min, point[0])
                y_min = min(y_min, point[1])
            else:
                x_max = max(x_max, point[0][0])
                y_max = max(y_max, point[0][1])
                x_min = min(x_min, point[0][0])
                y_min = min(y_min, point[0][1])
        image_part = img[y_min:y_max, x_min:x_max]
        color_filtered = color_filter(image_part)
        ret.append(color_filtered)
    return ret


def color_filter(img):
    """
    Filtering image by color, eliminting non-white, non-black and non-gray pixels

    :param img: image to be processed
    :return: processed image in grayscale format
    """
    res = img.copy()
    width, height, channel = img.shape
    mask = np.zeros((width, height))

    for i in range(width):
        for j in range(height):
            pixel = img[i, j]
            minn = min(pixel)
            maxx = max(pixel)
            if (maxx-minn) <= 40 and maxx < 160:
                mask[i, j] = 1
            else:
                res[i, j] = 255

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res


def process_plate_image(img):
    """
    Final image processing of the license plate image crop

    :param img: Plate image to process
    :return: Processed cv2::Mat image
    """

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.bitwise_not(img, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveBilateralFilter(gray, (7, 7), 15)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    show_image(thresh, 'Process 1 White')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (0, 0, 0)
    upper = (0, 0, 255)

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    # ret,gray = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    show_image(output, 'Process 2 White')
    return output