import cv2
from matplotlib import pyplot as plt


def show_image(image, title='image'):
    """
    Show the cv2::Mat image in a window and waits for a key

    :param image: Image to be shown
    :param title: Optional title for the window
    """

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
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
    cv2.drawContours(src2, rectangles, -1, (0, 255, 0), 2)
    show_image(src2)