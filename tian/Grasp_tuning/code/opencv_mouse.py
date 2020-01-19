import cv2
import numpy as np
import math
from visulize_trajectory import plot_grasp_path

drawing = False


def draw_circle(event, x, y, flags, param):
    global x1, y1, drawing, radius, num, img, img2
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
        radius = int(math.hypot(x - x1, y - y1))
        cv2.circle(img, (x1, y1), radius, (255, 0, 0), 1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            a, b = x, y
            if a != x & b != y:
                img = img2.copy()
                radius = int(math.hypot(a - x1, b - y1))
                cv2.circle(img, (x1, y1), radius, (255, 0, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        num += 1
        radius = int(math.hypot(x - x1, y - y1))
        cv2.circle(img, (x1, y1), radius, (255, 0, 255), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '_'.join(['label', str(num)]), (x + 20, y + 20), font, 1, (200, 255, 155), 1, cv2.LINE_AA)
        img2 = img.copy()


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, drawing, radius, num, img, img2
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
        angle = np.rad2deg(np.arctan2(y - y1, x - x1))
        plot_grasp_path([y1, x1], angle, 19, 130, img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            a, b = x, y
            if a != x & b != y:
                img = img2.copy()
                angle = np.rad2deg(np.arctan2(y - y1, x - x1))
                plot_grasp_path([y1, x1], angle, 19, 130, img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        num += 1
        angle = np.rad2deg(np.arctan2(y - y1, x - x1))
        plot_grasp_path([y1, x1], angle, 19, 130, img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '_'.join(['label', str(num)]), (x + 20, y + 20), font, 1, (200, 255, 155), 1, cv2.LINE_AA)
        img2 = img.copy()


if __name__ == "__main__":
    num = 0
    windowName = 'Drawing'

    img = np.zeros((500, 500, 3), np.uint8)
    img2 = img.copy()
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, draw_rectangle)
    while (True):
        cv2.imshow(windowName, img)
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyAllWindows()
