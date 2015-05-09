import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(image, image_label="", image_title='image', resize=True):
    """
    Show the cv2::Mat image in a window and wait for a key

    :type image: numpy.array
    :param image: Image to be shown
    :type image_label: str
    :param image_label: Optional image filename
    :type image_title: str
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

    :type titles: list[str]
    :param titles: Titles for each image
    :type images: list[numpy.array]
    :param images: Images to display
    :type rows: int
    :param rows: Row count of the matrix
    :type cols: int
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


def display_rectangles(image, rectangles, color=(0, 255, 0)):
    """
    Draw convex rectangle contours on top of the given image

    :type image: numpy.array
    :param image: Image on top of which to paint the contours
    :type rectangles: list[numpy.array]
    :param rectangles: Convex rectangles to be painted
    :type color: (int, int, int)
    :param color: Optional color for the rectangles. Default is green color
    """

    src2 = image.copy()
    cv2.drawContours(src2, rectangles, -1, color, 2)
    show_image(src2)


def draw_contours(image, contours, image_label="", color=(0, 255, 0)):
    """
    Plot specified contours on the specified image

    :type image: numpy.array
    :param image: Image on which to plot
    :type contours: list[numpy.array]
    :param contours: Contours to be plotted
    :type image_label: str
    :param image_label: Optional label for the image
    :type color: (int, int, int)
    :param color: Optional color for the contours. Default is green color
    """

    src = image.copy()
    cv2.drawContours(src, contours, -1, color, 1)
    show_image(src, image_label, 'Contours')


def get_parts_of_image(img, rectangles):
    """
    Crops the detected rectangles from the image and tries to find any text

    :type img: numpy.array
    :param img: Source image from which to crop
    :type rectangles: list[numpy.array]
    :param rectangles: Rectangles representing crop areas
    :rtype: list[numpy.array]
    :return: Images representing the cropped areas
    """

    ret = []
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


def get_white_pixels(img, rectangles):
    """
    Crops the detected rectangles from the image and masks pixels that aren't black, white or gray

    :type img: numpy.array
    :param img: Source image from which to crop
    :type rectangles: list[numpy.array]
    :param rectangles: List of rectangles representing crop areas
    :rtype: list[numpy.array]
    :return: Images representing the cropped areas
    """

    ret = []
    parts = get_parts_of_image(img, rectangles)
    for image_part in parts:
        color_filtered = color_filter(image_part)
        ret.append(color_filtered)
    return ret


def color_filter(img):
    """
    Filtering image by color, masking every pixel except black, white or gray

    :type img: numpy.array
    :param img: image to be processed
    :rtype: numpy.array
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
