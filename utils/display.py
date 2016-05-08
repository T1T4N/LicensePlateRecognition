import cv2
import numpy as np


def show_image(image, image_label="", image_title='image', resize=True):
    """
    Show the image in a window and wait for a key

    :type image: numpy.array
    :param image: Image to be shown
    :type image_label: str
    :param image_label: Optional image filename
    :type image_title: str
    :param image_title: Optional title for the window
    :type resize: bool
    :param resize: Optional flag to resize the output window
    """

    title = image_title + " " + image_label
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Create the window
    cv2.imshow(title, image)
    if resize:
        cv2.resizeWindow(title, 1050, 615)
    cv2.waitKey(0)  # Wait for a keypress
    cv2.destroyWindow(title)


def multi_plot(height, width, rows, cols, images, titles,
               title_color=(255, 255, 255), center_title=False,
               background_color=(0, 0, 0),
               padding=10, centered=True, borders=False, border_color=(0, 255, 0)):
    """
    Plot the specified images with the specified titles in a matrix of images

    :type height: int
    :param height: Height of the output image
    :type width: int
    :param width: Width of the output image
    :type rows: int
    :param rows: Row count of the matrix
    :type cols: int
    :param cols: Column count of the matrix
    :type images: list[numpy.array]
    :param images: List of images to be displayed
    :type titles: list[string]
    :param titles: Titles corresponding to each image
    :raises: ValueError if number of titles is not equal with number of images
    """

    if len(images) != len(titles):
        raise ValueError("The number of titles is different from the number of images")
    text_size, baseline = cv2.getTextSize("ABCDEFGH", cv2.FONT_HERSHEY_PLAIN, 1, 1)
    title_height = text_size[1]
    disp_image = np.zeros((height, width, 3), np.uint8)
    disp_image[:, :] = background_color
    piece_height = int(float(height - (rows + 1) * (padding + title_height)) / rows)
    piece_width = int(float(width - (cols + 1) * padding) / cols)
    xpos, ypos = padding, padding
    for i in range(rows):
        ypos = padding
        for j in range(cols):
            if i * cols + j >= len(images):
                break
            curr_img = images[i * cols + j]
            curr_title = titles[i * cols + j]
            img_height, img_width = curr_img.shape[0], curr_img.shape[1]
            scale = min(float(piece_width) / img_width, float(piece_height) / img_height)
            new_height = int(img_height * scale)
            new_width = int(img_width * scale)

            h_diff = int(abs(piece_height - new_height) / 2) if centered else 0
            w_diff = int(abs(piece_width - new_width) / 2) if centered else 0
            # cvSetImageROI(disp_image, cvRect(...)));
            disp_image[
            xpos + h_diff + title_height:xpos + h_diff + title_height + new_height,
            ypos + w_diff:ypos + w_diff + new_width
            ] = cv2.resize(curr_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            text_size, baseline = cv2.getTextSize(curr_title, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            t_diff = int(abs(piece_width - text_size[0]) / 2) if center_title else 0
            # disp_image[xpos, ypos+w_diff:ypos+w_diff+new_width] = (255, 0, 0)
            cv2.putText(disp_image, curr_title, (ypos + w_diff + t_diff, xpos + title_height - 1),
                        cv2.FONT_HERSHEY_PLAIN, 1, title_color, 1)

            if borders:
                # Horizontal
                disp_image[xpos + title_height, ypos:ypos + piece_width] = border_color
                disp_image[((i + 1) * (piece_height + padding + title_height)) - 1,
                ypos:ypos + piece_width] = border_color
                # Vertical
                disp_image[xpos + title_height:xpos + title_height + piece_height, ypos] = border_color
                disp_image[xpos + title_height:xpos + title_height + piece_height,
                ((j + 1) * piece_width) - 1 + (j + 1) * padding] = border_color
            ypos += piece_width + padding
        xpos += piece_height + padding + title_height
    return disp_image


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

        # Get the part of the image
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
            if (maxx - minn) <= 40 and maxx < 160:
                mask[i, j] = 1
            else:
                res[i, j] = 255

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res
