import os
import cv2
import numpy as np
from visulize_trajectory import plot_grasp_path
from helpers import extract_contact_region, cv_plot_contact_region, extract_contact_points, grasp_wrench_sum


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, angle, position_change, angle_change, img, img2, Im
    if event == cv2.EVENT_LBUTTONDOWN:
        position_change = True
        x1 = x
        y1 = y
    if event == cv2.EVENT_LBUTTONUP:
        position_change = False

    if event == cv2.EVENT_RBUTTONDOWN:
        angle_change = True
        if x1 == -1:
            x1 = x
            y1 = y
    if event == cv2.EVENT_RBUTTONUP:
        angle_change = False

    if event == cv2.EVENT_MOUSEMOVE:
        if position_change:
            x1, y1 = x, y
        if angle_change:
            angle = np.rad2deg(np.arctan2(y - y1, x - x1))
        if position_change or angle_change:
            img = img2.copy()
            contact_r, centering_score = extract_contact_region(y1, x1, angle, 19, 130, Im[:, :, 0])
            contact_ps, n1, n2 = extract_contact_points(contact_r, 8)
            torque = grasp_wrench_sum(contact_ps, n1, n2)
            cv_plot_contact_region(contact_r, img)
            plot_grasp_path([y1, x1], angle, 19, 130, img)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f'approximate torque score: [{torque[0]:.4f}, {torque[1]:.4f}],{n1, n2}'
            cv2.putText(img, text, (50, 100), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    img_rgb = 'wbt.png'
    img_mask = 'wbt_mask.png'
    I = cv2.imread(os.path.join(path, 'pictures', img_rgb))
    Im = cv2.imread(os.path.join(path, 'pictures', img_mask))

    windowName = 'Draw grasp rectangle'
    img = Im * 255
    img2 = img.copy()
    position_change = False
    angle_change = False
    x1 = -1
    y1 = -1
    angle = 0
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, draw_rectangle)
    while True:
        cv2.imshow(windowName, img)
        if cv2.waitKey(20) == ord('q'):
            break
    cv2.destroyAllWindows()
