import cv2
import numpy as np
import math
from utils import display, image


def deskew_lines(plate):
    angle_rad = 0.0
    img = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)

    # hq2x algorithm
    # img2x = image.hq2x_zoom(img)

    display.show_image(img, resize=True)
    # display.show_image(img2x, resize=True)

    img = plate.copy()  # cv2.cvtColor(img2x, cv2.COLOR_BGR2GRAY)
    disp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=3 * width / 4, maxLineGap=20)
    if lines is not None and len(lines) > 0:
        for i in range(len(lines[0])):
            line = lines[0, i]
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            cv2.line(disp_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
            line_angle = math.atan2(y2 - y1, x2 - x1)
            angle_rad += line_angle
            # print "Line angle: %.3f" % line_angle

        angle_rad /= len(lines[0])
        # print "Avg angle rad: %.3f" % angle_rad
        angle = math.degrees(angle_rad)
        print "Avg angle deg: %.3f\n" % angle

        display.show_image(disp_img, resize=True)
        rotation_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        disp_img = cv2.warpAffine(disp_img, rotation_mat, (width, height))
        display.show_image(disp_img, resize=True)
        img = cv2.warpAffine(img, rotation_mat, (width, height))

    return img


def deskew_text(plate):
    img = plate.copy()
    disp_img = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)
    img_height, img_width = img.shape
    img_area = img_height * img_width
    points = set([])
    boxes = []
    print "Deskewing text"
    point_to_rect = {}

    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for ct in contours:
        mr = cv2.minAreaRect(ct)  # Minimum enclosing rectangle
        box = cv2.cv.BoxPoints(mr)
        box = np.int0(box)
        box_points = [(box[i, 0], box[i, 1]) for i in range(len(box))]
        box_width, box_height = image.calculate_size(box_points)
        if box_width > 0 and box_height > 0:
            box_area = box_width * box_height
            box_ratio = box_width / box_height
            if box_ratio < 1:
                box_ratio = 1 / box_ratio

            print "Box width: %.3f, height: %.3f" % (box_width, box_height)
            print "Box area: %.3f" % box_area
            print "Box ratio: %.3f" % box_ratio
            print "Image area / box_area: %.3f" % (img_area / box_area)

            # TODO: Adjust img/box area ratio
            limit_area = 58
            limit_ratio = 4.1
            if 0.5 < box_ratio < limit_ratio and 4 < img_area / box_area < limit_area:
                print "Passed\n"

                cv2.drawContours(disp_img, [box], 0, (0, 255, 0), 1)
                boxes.append(box_points)
                for (x, y) in box_points:
                    points.add((x, y))
                    point_to_rect[(x, y)] = (mr, ct)
            else:
                cv2.drawContours(disp_img, [box], 0, (0, 0, 255), 1)
                # display.show_image(disp_img)

    points_sorted = sorted(points, key=lambda item: (item[0], -item[1]))
    if len(points_sorted) > 0:
        points_rev = points_sorted[::-1]

        border_margin = 3  # Adding a border margin to have a space of few pixels away from the edge

        bottom_left = points_sorted[0] if points_sorted[0][1] > points_sorted[1][1] else points_sorted[1]
        bottom_right = points_rev[0] if points_rev[0][1] > points_rev[1][1] else points_rev[1]

        leftmost = [point for point in points_sorted[:4] if point != bottom_left]
        rightmost = [point for point in points_rev[:4] if point != bottom_right]
        dl = {}
        dr = {}
        dist_left = []
        dist_right = []
        for point in leftmost:
            dst = abs(bottom_left[0] - point[0])
            dl[dst] = point
            dist_left.append(dst)
        for point in rightmost:
            dst = abs(bottom_right[0] - point[0])
            dr[dst] = point
            dist_right.append(dst)
        dist_left = sorted(dist_left)
        dist_right = sorted(dist_right)
        top_left = dl[dist_left[0]]
        top_right = dr[dist_right[0]]

        min_rect, contour = point_to_rect[bottom_left]
        min_rect2, contour2 = point_to_rect[bottom_right]
        angle1, angle2 = min_rect[2], min_rect2[2]

        # BUGFIX: When the first or last box contains J (for example) the minimum area bounding rectangle is rotated
        # Detect the differences between the rotation angles and use the non rotated bounding rectangle
        # for the box that has bigger angle
        if angle1 != 0.0 or angle2 != 0.0:
            a1 = angle1 % 90
            a2 = angle2 % 90
            diff = abs(abs(a1) - abs(a2))
            print a1, a2, diff

            if diff > 10 and a1 < a2:
                x, y, box_width, box_height = cv2.boundingRect(contour2)
                bottom_right = (x + box_width, y + box_height)
                top_right = (x + box_width, y)
                cv2.rectangle(disp_img, (x, y), (x + box_width, y + box_height), (255, 255, 0), 1)
            elif diff > 10 and a2 < a1:
                x, y, box_width, box_height = cv2.boundingRect(contour)
                bottom_left = (x, y + box_height)
                top_left = (x, y)
                cv2.rectangle(disp_img, (x, y), (x + box_width, y + box_height), (255, 255, 0), 1)

        bottom_left = (bottom_left[0] - border_margin, bottom_left[1] + border_margin)
        top_left = (top_left[0] - border_margin, top_left[1] - border_margin)
        bottom_right = (bottom_right[0] + border_margin, bottom_right[1] + border_margin)
        top_right = (top_right[0] + border_margin, top_right[1] - border_margin)

        corners = np.array([bottom_left, bottom_right, top_left, top_right], np.float32)
        dest_points = np.array([(0, img_height), (img_width, img_height), (0, 0), (img_width, 0)], np.float32)
        trans_matrix = cv2.getPerspectiveTransform(corners, dest_points)
        disp_wrapped = cv2.warpPerspective(img, trans_matrix, (img_width, img_height))

        # Translate each box coordinate to the new wrapped image
        # for box in boxes:
        # for x, y in box:
        #         xn = ((trans_matrix[0][0]*x+trans_matrix[0][1]*y+trans_matrix[0][2]) /
        #               (trans_matrix[2][0]*x+trans_matrix[2][1]*y+trans_matrix[2][2]))
        #         yn = ((trans_matrix[1][0]*x+trans_matrix[1][1]*y+trans_matrix[1][2]) /
        #               (trans_matrix[2][0]*x+trans_matrix[2][1]*y+trans_matrix[2][2]))
        #         cv2.circle(disp_wrapped, (int(xn), int(yn)), 1, (255, 0, 255), thickness=2)

        cv2.circle(disp_img, bottom_left, 1, (255, 0, 0), thickness=2)
        cv2.circle(disp_img, top_left, 1, (255, 0, 0), thickness=2)
        cv2.circle(disp_img, bottom_right, 1, (255, 0, 0), thickness=2)
        cv2.circle(disp_img, top_right, 1, (255, 0, 0), thickness=2)

        display.show_image(disp_img)
        display.show_image(disp_wrapped)
        return disp_wrapped

    return img