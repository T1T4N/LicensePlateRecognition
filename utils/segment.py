import cv2
import numpy as np
import display
from recognizer import TextRecognizer


def segment_contours(plate):
    img = plate.copy()
    img2 = img.copy()
    disp_img = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)
    img_height, img_width = img.shape
    img_area = img_height * img_width

    print "\n\nSegmenting contours\nPart area: %.3f" % img_area

    boxes = []
    # RETR_CCOMP returns a two level hierarchy of the contours: parent and children
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i, ct in enumerate(contours):
        # mr = cv2.minAreaRect(ct)
        # box = cv2.cv.BoxPoints(mr)
        # box = np.int0(box)
        # box_points = [(box[i, 0], box[i, 1]) for i in range(len(box))]
        # box_width, box_height = image.calculate_size(box_points)

        if hierarchy[0][i][3] != -1:
            continue

        # TODO: Number one has ratio ~ 4.5: very thin and high
        x, y, box_width, box_height = cv2.boundingRect(ct)
        if box_width > 0 and box_height > 0:
            box_area = float(box_width) * float(box_height)

            box_ratio = float(box_width) / float(box_height)
            if box_ratio < 1:
                box_ratio = 1 / float(box_ratio)

            print "Box width: %.3f, height: %.3f" % (box_width, box_height)
            print "Box area: %.3f" % box_area
            print "Box ratio: %.3f" % box_ratio
            print "Area ratio: %.3f" % (img_area / box_area)

            # TODO: Square in the middle always caught, adjust box_ratio upper limit
            if 0.5 < box_ratio < 3 and img_area / box_area < 45:
                print "Passed\n"

                # TODO: Fill contour without the holes
                # cv2.drawContours(img, [ct], 0, (255, 255, 255), thickness=-1)

                # cv2.drawContours(disp_img, [box], 0, (0, 0, 255), 1)
                cv2.rectangle(disp_img, (x, y), (x + box_width, y + box_height), (0, 255, 0), 1)

                box_points = np.array(
                    [(x, y), (x, y + box_height), (x + box_width, y), (x + box_width, y + box_height)]
                )
                # box_points = sorted(box_points, key=lambda item: (item[0], item[1]))
                boxes.append(np.array(box_points))

    # Fill holes of contours in black
    for i, ct in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            parent_idx = hierarchy[0][i][3]
            parent_contour = contours[parent_idx]
            parent_area = cv2.contourArea(parent_contour)
            child_area = cv2.contourArea(ct)
            if child_area > float(parent_area) / 9:
                # Approximate using a polygon
                peri = cv2.arcLength(ct, True)
                approx = cv2.approxPolyDP(ct, 0.020 * peri, True)
                if cv2.isContourConvex(approx):
                    # cv2.drawContours(img, [approx], 0, (0, 0, 0), thickness=-1)
                    pass

    boxes_sorted = sorted(boxes, key=lambda item: (item[0][0], item[0][1]))
    boxes_sep = display.get_parts_of_image(img, boxes_sorted)
    display.show_image(disp_img)

    labels = []
    for box in boxes_sep:
        # box_mod = image.hq2x_zoom(cv2.cvtColor(box, cv2.COLOR_GRAY2BGR))
        box_mod = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
        tr = TextRecognizer(cv2.bitwise_not(box_mod))
        text, conf = tr.find_text()
        text = text.strip()
        t2 = ""
        for i in range(len(text)):
            if ord(text[i]) in range(128):
                t2 += text[i]
        labels.append(t2)
        print t2, conf
        # display.show_image(box_mod)

    display.multi_plot(boxes_sep, labels, 1, len(boxes_sep))
