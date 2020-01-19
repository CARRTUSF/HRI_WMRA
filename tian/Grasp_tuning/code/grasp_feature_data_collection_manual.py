import os
import cv2
import numpy as np
from visulize_trajectory import plot_grasp_path
from helpers import extract_contact_region, cv_plot_contact_region, high_level_grasp_feature


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, angle, position_change, angle_change, img, img2, Im, data_count
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
            contact_rl, contact_rr = extract_contact_region(y1, x1, angle, 19, 130, Im[:, :, 0])
            cli, trans, rot, slip, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr, angle, 130)
            cv_plot_contact_region(contact_rl, img, True, lcids)
            cv_plot_contact_region(contact_rr, img, True, rcids)
            # contact_ps, n1, n2 = extract_contact_points(contact_r, 8)
            # cv_plot_contact_region(contact_ps, img, False)
            plot_grasp_path([y1, x1], angle, 19, 130, img)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f'{data_count}'
            cv2.putText(img, text, (100, 100), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


def group_data(a, width):
    n = a.shape[0]
    a0s = np.ndarray((0, 2), dtype=np.float16)  # [[avg, number of items]]
    group = np.ndarray((n, 2), dtype=np.float16)
    for i in range(n):
        group[i, 0] = a[i]
        group_id = -1
        total_groups = a0s.shape[0]
        for j in range(total_groups):
            if abs(a[i] - a0s[j, 0]) <= width:
                group_id = j
                break
        if group_id != -1:
            group[i, 1] = group_id
            a0s[group_id, 0] = (a0s[group_id, 0] * a0s[group_id, 1] + a[i]) / (a0s[group_id, 1] + 1)
            a0s[group_id, 1] += 1
        else:
            group[i, 1] = total_groups
            a0s = np.append(a0s, [[a[i], 1]], axis=0)
    return a0s, group


def select_dominant_minimum(data_groups, dominant_threshold):
    number_of_groups = data_groups.shape[0]
    total_data = np.sum(data_groups[:, 1])
    main_groups = np.ndarray((0, 3), dtype=np.float16)
    dominant_groups = np.ndarray((0, 3), dtype=np.float16)
    for i in range(number_of_groups):
        if data_groups[i, 1] / total_data >= 1 / number_of_groups:
            main_groups = np.append(main_groups, [[data_groups[i, 0], data_groups[i, 1], i]], axis=0)
    main_max = np.amax(main_groups[:, 1])
    for j in range(main_groups.shape[0]):
        if main_groups[j, 1] / main_max > dominant_threshold:
            dominant_groups = np.append(dominant_groups, [main_groups[j, :]], axis=0)
    dominant_minimum_group = np.amin(dominant_groups[:, 0])
    dominant_minimum_group_id = dominant_groups[np.argmin(dominant_groups[:, 0]), 2]
    return dominant_minimum_group, dominant_minimum_group_id


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    img_rgb = 'wbt.png'
    img_mask = 'spoon3_mask.png'
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
    data_count = 0
    data = np.ndarray((0, 3), dtype=np.float16)
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, draw_rectangle)
    while True:
        cv2.imshow(windowName, img)
        usr_input = cv2.waitKey(20)
        if usr_input == ord('q'):
            break
        elif usr_input == ord('s'):
            data_count += 1
            data_i = [y1, x1, angle]
            print('save data', data_count)
            data = np.append(data, [data_i], axis=0)
            if data_count == 100:
                break
    cv2.destroyAllWindows()
    np.savetxt(f'cd1.txt', data, fmt='%1.4f')

    # a1 = np.arange(50)
    # a2 = np.random.rand(50)*30
    # aos1, ag1 = group_data(a1, 5)
    # aos2, ag2 = group_data(a2, 5)
    # print(aos1)
    # print(ag1)
    # print('--------')
    # print(aos2)
    # print(ag2)
    #
    # min1 = select_dominant_minimum(aos1, 0.65)
    # min2 = select_dominant_minimum(aos2, 0.65)
    #
    # print(min1)
    # print('--------')
    # print(min2)
