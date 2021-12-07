import cv2
import numpy as np

def preprocess(img): #take an image and process it to values that python can work with more easily, values can be adjusted
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode


#takes in 2 lists, an approximate contour of each shape points have to be exactly 2 more than convex hull
def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)
    #In order to know whether you should subtract 2 from the first element of the indices list, or add 2,
    #you'll need to do the exact opposite to the second (which is the last) element of the indices list;
    #if the resulting two indices returns the same value from the points list, then you found the tip of the arrow.
    #I used a for loop that loops through numbers 0 and 1. The first iteration will add 2 to the second element of the indices list:
    #j = indices[i] + 2, and subtract 2 from the first element of the indices list: indices[i - 1] - 2:
    for i in range(2):
        j = indices[i] + 2
        #if you try adding 2 to the index 5, you will get an IndexError. So if, say j becomes 7 from the j = indices[i] + 2, the above condition will convert j to len(points) - j.
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]): #found the point of the arrow
            return tuple(points[j])

cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()

    contours, hierarchy = cv2.findContours(preprocess(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.047 * peri, True) #sensitivity
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        if 6 > sides > 3 and sides + 2 == len(approx):
            arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
            if arrow_tip:
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
                cv2.circle(img, arrow_tip, 3, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()