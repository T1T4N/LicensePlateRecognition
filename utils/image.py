__author__ = 'robert'

import math
import numpy as np

from PIL import Image
from hq2x import hq2x


def hq2x_zoom(source_image):
    """
    Performs a 2x zoom on a picture using the hqx algorithm

    :type source_image: numpy.array
    :param source_image: An image to be zoomed
    :rtype: np.array
    :return: The zoomed picture
    """

    source = Image.fromarray(source_image)
    dest = hq2x(source)
    return np.array(dest)


def calculate_size(points):
    """
    Given an array of four coordinates calculates width and height of the rectangle they form

    :type points: list[(int, int)]
    :param points: Four (x, y) tuples
    :rtype: (float, float)
    :return: A (width, height) tuple
    """

    # Sort points by their x-coordinate
    points = sorted(points, key=lambda item: (item[0], item[1]))

    h1 = math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1])
    h2 = math.hypot(points[2][0] - points[3][0], points[2][1] - points[3][1])
    height = max(h1, h2)

    d11 = abs(points[0][1] - points[2][1])
    d12 = abs(points[0][1] - points[3][1])
    d21 = abs(points[1][1] - points[2][1])
    d22 = abs(points[1][1] - points[3][1])

    width = -1
    if d11 < d12:
        width = max(width,
                    math.hypot(points[0][0] - points[2][0], points[0][1] - points[2][1]))
    else:
        width = max(width,
                    math.hypot(points[0][0] - points[3][0], points[0][1] - points[3][1]))

    if d21 < d22:
        width = max(width,
                    math.hypot(points[1][0] - points[2][0], points[1][1] - points[2][1]))
    else:
        width = max(width,
                    math.hypot(points[1][0] - points[3][0], points[1][1] - points[3][1]))

    return width, height
