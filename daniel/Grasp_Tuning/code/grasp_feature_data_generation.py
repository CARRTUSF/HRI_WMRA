import os
import cv2
import numpy as np
from visulize_trajectory import plot_grasp_path
from helpers import extract_contact_region, grasp_quality_measure, extract_contact_points, contact_angles, \
    grasp_torque_feature


def generate_grasp_sample(r1, c1, r2, c2, sample_size):
    rows = np.random.randint(r1, r2, size=sample_size).reshape((-1, 1))
    cols = np.random.randint(c1, c2, size=sample_size).reshape((-1, 1))
    angles = np.random.normal(0, 35, sample_size).reshape((-1, 1))
    grasp_centers = np.concatenate((rows, cols, angles), axis=1)
    return grasp_centers


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, position_change, img, img2
    if event == cv2.EVENT_LBUTTONDOWN:
        position_change = True
        x1 = x
        y1 = y
    if event == cv2.EVENT_LBUTTONUP:
        position_change = False
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'press e to confirm selection, or re-draw'
        cv2.putText(img, text, (50, 100), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    if event == cv2.EVENT_MOUSEMOVE:
        if position_change:
            x2, y2 = x, y
            img = img2.copy()
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0))


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    img_rgb = 'wbt.png'
    img_mask = 'wbt_mask.png'
    I = cv2.imread(os.path.join(path, 'pictures', img_rgb))
    Im = cv2.imread(os.path.join(path, 'pictures', img_mask))

    collected_data = np.ndarray((0, 12), dtype=np.float16)
    windowName = 'grasp data sample generation'
    img = Im * 255
    img2 = img.copy()
    position_change = False
    x1 = -1
    y1 = -1
    x2 = -1
    y2 = -1
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, draw_rectangle)
    while True:
        cv2.imshow(windowName, img)
        if cv2.waitKey(20) == ord('e'):
            break
    # cv2.destroyAllWindows()
    print(x1, y1, "->", x2, y2)
    sample = generate_grasp_sample(y1, x1, y2, x2, 500)
    for point in sample:
        cv2.circle(img, (int(point[1]), int(point[0])), 1, (0, 0, 255))
    cv2.imshow('grasp samples', img)
    confirmed = cv2.waitKey(0)
    if confirmed == ord("q"):
        cv2.destroyAllWindows()
    elif confirmed == ord('e'):
        for point in sample:
            contact_r, centering_score, single_contact_lines, gpl_center = extract_contact_region(point[0], point[1],
                                                                                                  point[2], 19, 130,
                                                                                                  Im[:, :, 0])
            contact_ps, n1, n2 = extract_contact_points(contact_r, 8)
            torque = grasp_torque_feature(contact_ps, n1, n2)
            ang_l, ang_r, ang_v, n_sum = contact_angles(contact_r, point[2])
            score, s1, s2, s3, sm_w = grasp_quality_measure(abs(centering_score), torque, ang_l + ang_r,
                                                            single_contact_lines)
            data = [score, centering_score, gpl_center, torque[0], torque[1], ang_l, ang_r, ang_v, n_sum, point[0],
                    point[1], point[2]]
            collected_data = np.append(collected_data, [data], axis=0)
        np.savetxt(f'unranked_grasp_data.txt', collected_data, fmt='%1.4f')
        cv2.destroyAllWindows()
