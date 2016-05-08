import cv2
import numpy as np
import display


def segment_contours(plate):
    """
    Finds the contours satisfying the required constraints

    :type plate: numpy.array
    :param plate: A gray image of the license plate
    :rtype: list[numpy.array]
    :return: BGR images of the contours
    """

    img = plate.copy()
    # The below copy is used only to visualize the process
    disp_img = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)

    img_height, img_width = img.shape
    img_area = img_height * img_width

    if __debug__:
        print("\nSegmenting contours\nPart area: %.3f" % img_area)

    # Filter small noise points by filling them with black color
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, ct in enumerate(contours):
        x, y, box_width, box_height = cv2.boundingRect(ct)
        if box_width > 0 and box_height > 0:
            box_area = float(box_width) * float(box_height)
            box_ratio = float(box_width) / float(box_height)
            if box_ratio < 1:
                box_ratio = 1 / float(box_ratio)

            limit_ratio = 5.5
            limit_area = 45.0
            if not (box_ratio < limit_ratio
                    and box_height / float(box_width) < limit_ratio
                    and img_area / box_area < limit_area
            ) and float(img_area) / box_area > limit_area:
                cv2.drawContours(img, [ct], 0, (0, 0, 0), thickness=-1)
                cv2.drawContours(disp_img, [ct], 0, (0, 0, 0), thickness=-1)

    boxes = []
    # Find the contours satisfying the conditions i.e the license plate characters
    # RETR_TREE returns a hierarchy of the contours: parent and children
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, ct in enumerate(contours):

        # Skip the contour if it has a parent contour
        if hierarchy[0][i][3] != -1:
            continue

        x, y, box_width, box_height = cv2.boundingRect(ct)
        if box_width > 0 and box_height > 0:
            box_area = float(box_width) * float(box_height)
            box_ratio = float(box_width) / float(box_height)
            if box_ratio < 1:
                box_ratio = 1 / float(box_ratio)

            # TODO: Square in the middle always caught, adjust box_ratio upper limit
            # TODO: Number 1 (one) has ratio ~ 4.5: very thin and high
            limit_ratio = 5.5
            limit_area = 45.0
            if box_ratio < limit_ratio and box_height / float(box_width) < limit_ratio \
                    and 4 < img_area / box_area < limit_area:
                if __debug__:
                    print("Box width: %.3f, height: %.3f" % (box_width, box_height))
                    print("Box area: %.3f" % box_area)
                    print("Box ratio: %.3f" % box_ratio)
                    print("Area ratio: %.3f" % (img_area / box_area))
                    print("Passed\n")

                # Experimental: fill contour with color
                # cv2.drawContours(img, [ct], 0, (255, 255, 255), thickness=-1)

                # Draw a rectangle around the contour (for visualization only)
                cv2.rectangle(disp_img, (x, y), (x + box_width, y + box_height), (0, 255, 0), 1)

                box_points = np.array(
                    [(x, y), (x, y + box_height), (x + box_width, y), (x + box_width, y + box_height)]
                )
                boxes.append(np.array(box_points))
            else:
                # Once again filter small noise points by filling them with black color
                # in case some were missed the first time
                cv2.rectangle(disp_img, (x, y), (x + box_width, y + box_height), (0, 0, 255), 1)
                if img_area / box_area > limit_area:
                    cv2.drawContours(img, [ct], 0, (0, 0, 0), thickness=-1)
                    cv2.drawContours(disp_img, [ct], 0, (0, 0, 0), thickness=-1)

    # EXPERIMENTAL
    # The idea is to first fill a contour with a solid color
    # and after that fill any child contour with black color

    # for i, ct in enumerate(contours):
    # if hierarchy[0][i][3] != -1:
    #         parent_idx = hierarchy[0][i][3]
    #         parent_contour = contours[parent_idx]
    #         parent_area = cv2.contourArea(parent_contour)
    #         child_area = cv2.contourArea(ct)
    #         if child_area > float(parent_area) / 9:
    #             # Approximate using a polygon
    #             peri = cv2.arcLength(ct, True)
    #             approx = cv2.approxPolyDP(ct, 0.020 * peri, True)
    #             if cv2.isContourConvex(approx):
    #                 cv2.drawContours(img, [approx], 0, (0, 0, 0), thickness=-1)

    # sort the arrays representing the boxes by their x-coordinate
    boxes_sorted = sorted(boxes, key=lambda item: (item[0][0], item[0][1]))
    boxes_sep = display.get_parts_of_image(img, boxes_sorted)
    if __debug__:
        display.show_image(disp_img, resize=False)

    return [cv2.cvtColor(box, cv2.COLOR_GRAY2BGR) for box in boxes_sep]
