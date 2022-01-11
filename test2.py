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
<<<<<<< HEAD
=======
    print(arrow_angle)
>>>>>>> parent of 77d48fc (cleaned stuff up)
    return arrow_angle


def get_arrow_info(arrow_image):
    arrow_info_image = cv2.cvtColor(arrow_image.copy(), cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(arrow_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrow_info = []
    if hierarchy is not None:

        for cnt in contours:
            # draw single arrow on blank image
            blank_image = np.zeros_like(arrow_image)
            cv2.drawContours(blank_image, [cnt], -1, 255, -1)

            point1, point2 = get_max_distace_point(cnt)
            angle = angle_beween_points(point1, point2)
            lenght = get_length(point1, point2)

            cv2.line(arrow_info_image, point1, point2, (0, 255, 255), 1)

            cv2.circle(arrow_info_image, point1, 2, (255, 0, 0), 3)
            cv2.circle(arrow_info_image, point2, 2, (255, 0, 0), 3)

            cv2.putText(arrow_info_image, "angle : {0:0.2f}".format(angle),
                        point2, cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
            cv2.putText(arrow_info_image, "lenght : {0:0.2f}".format(lenght),
                        (point2[0], point2[1] + 20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
                        
        return arrow_info_image, arrow_info
    else:
        return None, None


if __name__ == "__main__":
    while True:
<<<<<<< HEAD
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        #image = cv2.imread("FHtLk.png")
        
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
=======
        #cap = cv2.VideoCapture(0)
        #ret, frame = cap.read()
        image = cv2.imread("i.png")
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
>>>>>>> parent of 77d48fc (cleaned stuff up)
        _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("thresh_image", thresh_image)

        arrow_image = get_filter_arrow_image(thresh_image)
<<<<<<< HEAD
=======
        print(arrow_image)
>>>>>>> parent of 77d48fc (cleaned stuff up)
        print("b")
        if arrow_image is not None:
            cv2.imshow("arrow_image", arrow_image)

            arrow_info_image, arrow_info = get_arrow_info(arrow_image)
<<<<<<< HEAD
            cv2.imshow("arrow_info_image", arrow_info_image)
            print ("a")
        print ("s")

        cap.release()

        cv2.imshow("Image", frame)
=======
            #cv2.imshow("arrow_info_image", arrow_info_image)
            print ("a")
        print ("s")

        #cap.release()

        cv2.imshow("Image", image)
>>>>>>> parent of 77d48fc (cleaned stuff up)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()