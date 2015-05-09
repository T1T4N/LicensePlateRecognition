__author__ = 'robert'

import math
import numpy as np

from PIL import Image
from hq2x import hq2x


def hq2x_zoom(source_image):
    """
    Performs a 2x zoom on a picture using the hqx algorithm
    :type source_image: numpy.array representing an image
    :return: np.array representing the zoomed picture
    """
    source = Image.fromarray(source_image)
    dest = hq2x(source)
    return np.array(dest)


def calculate_size(coordinates):
    """
    Given an array of four coordinates calculates width and height of the rectangle they form

    :param coordinates: An array of 4 (x, y) tuples
    :return: A (width, height) tuple
    """

    sorted_coords = sorted(coordinates, key=lambda item: (item[0], item[1]))
    h1 = math.hypot(sorted_coords[0][0] - sorted_coords[1][0], sorted_coords[0][1] - sorted_coords[1][1])
    h2 = math.hypot(sorted_coords[2][0] - sorted_coords[3][0], sorted_coords[2][1] - sorted_coords[3][1])
    height = max(h1, h2)

    d11 = abs(sorted_coords[0][1] - sorted_coords[2][1])
    d12 = abs(sorted_coords[0][1] - sorted_coords[3][1])

    d21 = abs(sorted_coords[1][1] - sorted_coords[2][1])
    d22 = abs(sorted_coords[1][1] - sorted_coords[3][1])

    width = -1
    if d11 < d12:
        width = max(width,
                    math.hypot(sorted_coords[0][0] - sorted_coords[2][0], sorted_coords[0][1] - sorted_coords[2][1]))
    else:
        width = max(width,
                    math.hypot(sorted_coords[0][0] - sorted_coords[3][0], sorted_coords[0][1] - sorted_coords[3][1]))

    if d21 < d22:
        width = max(width,
                    math.hypot(sorted_coords[1][0] - sorted_coords[2][0], sorted_coords[1][1] - sorted_coords[2][1]))
    else:
        width = max(width,
                    math.hypot(sorted_coords[1][0] - sorted_coords[3][0], sorted_coords[1][1] - sorted_coords[3][1]))

    return width, height
