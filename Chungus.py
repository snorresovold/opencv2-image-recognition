import math
import cv2
import numpy as np

def get_filter_arrow_image(threslold_image):
    blank_image = np.zeros_like(threslold_image)

    # dilate image to remove self-intersections error
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    threslold_image = cv2.dilate(threslold_image, kernel_dilate, iterations=1)

    contours, hierarchy = cv2.findContours(threslold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:

        threshold_distnace = 1000

        for cnt in contours:
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    start_index, end_index, farthest_index, distance = defects[i, 0]

                    # you can add more filteration based on this start, end and far point
                    # start = tuple(cnt[start_index][0])
                    # end = tuple(cnt[end_index][0])
                    # far = tuple(cnt[farthest_index][0])

                    if distance > threshold_distnace:
                        cv2.drawContours(blank_image, [cnt], -1, 255, -1)

        return blank_image
    else:
        return None


def get_length(p1, p2):
    line_length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return line_length


def get_max_distace_point(cnt):
    max_distance = 0
    max_points = None
    for [[x1, y1]] in cnt:
        for [[x2, y2]] in cnt:
            distance = get_length((x1, y1), (x2, y2))

            if distance > max_distance:
                max_distance = distance
                max_points = [(x1, y1), (x2, y2)]

    return max_points


def angle_beween_points(a, b):
    arrow_slope = (a[0] - b[0]) / (a[1] - b[1])
    arrow_angle = math.degrees(math.atan(arrow_slope))
    print(arrow_angle)
    return arrow_angle

