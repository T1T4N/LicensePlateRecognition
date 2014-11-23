import cv2
from matplotlib import pyplot as plt


def show_image(image, title='image'):
    """Shows the cv2::Mat image in a window and waits for a key"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
    #cv2.imwrite('test.jpg', src)


def multi_plot(titles, images, rows, cols):
    """Plots the specified images with the specified titles
    in a matrix of images with given row count and column count"""
    if len(titles) == len(images):
        for i in xrange(len(titles)):
            plt.subplot(rows, cols, i+1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


def display_rectangles(image, rectangles):
    """Draws the contours which are convex rectangles
     with green color on top of the given image
     :returns The biggest rectangle drawn, as a set of points"""
    src2 = image.copy()
    cv2.drawContours(src2, rectangles, -1, (0, 255, 0), 2)
    show_image(src2)