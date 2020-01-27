import os
import cv2
import numpy as np
from vgt2 import find_contact_region, get_outline_and_normal
from visulize_trajectory import plot_grasp_path
import time
from helpers import extract_contact_region, cv_plot_contact_region


def get_object_pixels(mask_img):
    object_pixels = []
    r, c = mask_img.shape
    for i in range(r):
        for j in range(c):
            if mask_img[i, j] == 1:
                object_pixels.append([i, j])
    return object_pixels


def grasp_features(contact_region, outline_pixels, normals, gripper_center, theta):  # gripper_center(row,col)
    """

    :param contact_region: nx[row,col,side]
    :param outline_pixels:
    :param normals: nx[row0,col0,row1,col1]
    :param gripper_center: [row,col]
    :param theta: gripper roll
    :return: x_loss, y_loss, theta_loss
    """
    normal_sum = np.array([0.0, 0.0])
    vector_sum = np.array([0.0, 0.0])
    left_sum = np.array([0.0, 0.0])
    right_sum = np.array([0.0, 0.0])
    gripper_center = np.array(gripper_center)
    left_count = 0
    right_count = 0
    lv = np.array([0.0, 0.0])
    rv = np.array([0.0, 0.0])
    # lc = np.array([0.0, 0.0])
    # rc = np.array([0.0, 0.0])
    if contact_region.shape[0] > 60:
        for contact_marker in contact_region:
            contact = outline_pixels[contact_marker[0]]
            vector_sum += contact - gripper_center  # [row, col]
            normal_sum += normals[contact_marker[0]]
            if contact_marker[1] == 0:
                # left side
                left_count += 1
                lv += contact - gripper_center
                # lc += contact
                left_sum += normals[contact_marker[0]]  # [row, col]
            else:
                # right side
                right_count += 1
                # rc += contact
                rv += contact - gripper_center
                right_sum += normals[contact_marker[0]]  # [row, col]
        x_loss = (vector_sum[0] ** 2 + vector_sum[1] ** 2) ** 0.5
        y_loss = (normal_sum[0] ** 2 + normal_sum[1] ** 2) ** 0.5
        new_vsum = lv / left_count + rv / right_count
        new_x_loss = (new_vsum[0] ** 2 + new_vsum[1] ** 2) ** 0.5
        c_v_ang = np.dot(left_sum, right_sum) / (
                    (left_sum[0] ** 2 + left_sum[1] ** 2) ** 0.5 * (right_sum[0] ** 2 + right_sum[1] ** 2) ** 0.5)
        v_ang = np.rad2deg(np.arccos(c_v_ang))
        # lcc = lc/left_count
        # rcc = rc/right_count
        # offset_loss = abs(rcc[0]-lcc[0])
        r_normal = np.array([np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))])  # [row, col]
        l_normal = np.array([-np.sin(np.deg2rad(theta)), -np.cos(np.deg2rad(theta))])  # [row, col]
        if all([v == 0 for v in left_sum]):
            alpha = 200
            beta = 300
        elif all([v == 0 for v in right_sum]):
            beta = 200
            alpha = 300
        else:
            c_alpha = np.dot(l_normal, left_sum) / (left_sum[0] ** 2 + left_sum[1] ** 2) ** 0.5
            c_beta = np.dot(r_normal, right_sum) / (right_sum[0] ** 2 + right_sum[1] ** 2) ** 0.5
            alpha = np.rad2deg(np.arccos(c_alpha))
            beta = np.rad2deg(np.arccos(c_beta))
        theta_loss_p = alpha + beta
        theta_loss_s = abs(alpha - beta)
        print('angles alpha, beta, vang', alpha, beta, v_ang)
        print('contact size left, right', left_count, right_count)
        return x_loss, y_loss, theta_loss_p, theta_loss_s, new_x_loss, v_ang
    else:
        return -1, -1, -1, -1, -1, -1


def get_score(k1, k2, k3, option):
    if option == 1:
        if k1 == -1:
            s1 = 0
        else:
            y = 4 * (1 - k1 / 5000)
            s1 = max(y, 0)
    else:
        if k1[0] == -1:
            s1 = 0
        else:
            # if k1[0] > 80:
            #     y1 = 2 - k1[0] / 80
            # else:
            #     y1 = 5 - (k1[0]/80 + 1)**2
            if k1[0] < 40:
                y1 = 4 - k1[0] / 40
            else:
                y1 = 4.5 - 3 * k1[0] / 80
            y2 = 4 * (1 - (180 - k1[1]) / 40)
            sk10 = max(y1, 0)
            sk11 = max(y2, 0)
            print('v3 centering score', sk10)
            print('v3 contact fit score', sk11)
            s1 = min(sk10, sk11)

    if k2 == -1:
        s2 = 0
    else:
        y = 4 * (1 - k2 / 32)
        s2 = max(y, 0)

    if k3 == -1:
        s3 = 0
    else:
        y = 4 * (1 - k3 / 72)
        s3 = max(y, 0)

    print('v3 scores', s1, s2, s3)
    s_min = min(s1, s2, s3)
    min_weight = max(0.95 - 0.15 * s_min, 0.5)
    print('smin, min_weight', s_min, min_weight)
    score = (1 - min_weight) * (s1 + s2 + s3) / 3 + min_weight * s_min
    return score


def get_score2(k1, k2, k3, option):
    if option == 1:
        if k1 == -1:
            s1 = 0
        else:
            y = 4 * (1 - k1 / 5000)
            s1 = max(y, 0)
    else:
        if k1[0] == -1:
            s1 = 0
        else:
            y1 = 4 * (1 - k1[0] / 160)
            y2 = 4 * (1 - (180 - k1[1]) / 40)
            sk10 = max(y1, 0)
            sk11 = max(y2, 0)
            print('v2 centering score', sk10)
            print('v2 contact fit score', sk11)
            s1 = min(sk10, sk11)

    if k2 == -1:
        s2 = 0
    else:
        y = 4 * (1 - k2 / 36)
        s2 = max(y, 0)

    if k3 == -1:
        s3 = 0
    else:
        y = 4 * (1 - k3 / 76)
        s3 = max(y, 0)

    print('v2 scores', s1, s2, s3)
    s_min = min(s1, s2, s3)
    score = s_min * ((s1 + s2 + s3) / 3 - s_min) / 8 + s_min
    return score


def get_score1(k1, k2, k3, option):
    if option == 1:
        if k1 >= 4000 or k1 == -1:
            s1 = 0
        elif 3000 <= k1 < 4000:
            s1 = 1
        elif 2000 <= k1 < 3000:
            s1 = 2
        elif 1000 <= k1 < 2000:
            s1 = 3
        else:
            s1 = 4
    else:
        if k1[0] >= 70 or k1[0] == -1:
            sk10 = 0
        elif 50 <= k1[0] < 70:
            sk10 = 1
        elif 30 <= k1[0] < 50:
            sk10 = 2
        elif 15 <= k1[0] < 30:
            sk10 = 3
        else:
            sk10 = 4

        if k1[1] >= 172:
            sk11 = 4
        elif 164 <= k1[1] < 172:
            sk11 = 3
        elif 156 <= k1[1] < 164:
            sk11 = 2
        elif 148 <= k1[1] < 156:
            sk11 = 1
        else:
            sk11 = 0
        print('v1 centering score', sk10)
        print('v1 contact fit score', sk11)
        s1 = min(sk10, sk11)
    if k2 >= 35 or k2 == -1:
        s2 = 0
    elif 25 <= k2 < 35:
        s2 = 1
    elif 15 <= k2 < 25:
        s2 = 2
    elif 5 <= k2 < 15:
        s2 = 3
    else:
        s2 = 4

    if k3 >= 75 or k3 == -1:
        s3 = 0
    elif 55 <= k3 < 75:
        s3 = 1
    elif 35 <= k3 < 55:
        s3 = 2
    elif 15 <= k3 < 35:
        s3 = 3
    else:
        s3 = 4

    print('v1 scores', s1, s2, s3)
    s_min = min(s1, s2, s3)
    score = s_min / 2 + (s1 + s2 + s3) / 6
    return score


# def gather_data(mask_img, obj_pt):
#     outline_pixels, outline_normals, index_array = get_outline_and_normal(mask_img[:, :, 0], 0, 0, 7)
#     results = np.ndarray((0, 11), dtype=np.float16)
#     interrupted = False
#     # plot data
#     # plt.ion()
#     # fig = plt.figure()
#     while not interrupted:
#         new_center_i = random.randint(0, len(obj_pt))
#         new_angle = random.uniform(-90.0, 90.0)
#         contact = find_contact_region(outline_pixels, index_array, obj_pt[new_center_i], new_angle)
#         l1, l2, l3, l4, l5, va, l6 = grasp_features(contact, outline_pixels, outline_normals, obj_pt[new_center_i], new_angle)
#         score1 = get_score(l1, l2, l3, 1)
#         score2 = get_score([l5, va], l2, l3, 2)
#         # score3 = get_score1(l1, l2, l3, 1)
#         # score4 = get_score1([l5, va], l2, l3, 2)
#         # score5 = get_score2(l1, l2, l3, 1)
#         # score6 = get_score2([l5, va], l2, l3, 2)
#         data = [0, score1, score2, obj_pt[new_center_i][0], obj_pt[new_center_i][1], new_angle, l5, l1, l2, l3, l4]  # current grasp data
#         print('current data', data)
#         print('old xloss', l1)
#         print('new xloss', l5)
#         print('offsetloss', l6)
#         # print('new scores', score3, score4)
#         # print('old scores', score5, score6)
#         img = np.copy(mask_img * 255)
#         plot_grasp_path(obj_pt[new_center_i], new_angle, 19, 130, img)
#         img_s = cv2.resize(img, (1024, 640))
#         cv2.imshow('choose the better grasp', img_s)
#         input_captured = False
#         while not input_captured:
#             usr_input = cv2.waitKey(0)
#             if usr_input == ord('q'):
#                 print('stopped recording')
#                 interrupted = True
#                 break
#             else:
#                 try:
#                     data[0] = int(chr(usr_input))
#                     print(f'the score you entered: {data[0]}')
#                     input_captured = True
#                 except ValueError:
#                     print('enter 0-5')
#         results = np.append(results, [data], axis=0)
#     return results


def calculate_data(r, c, ang, outline_pixels, index_array, outline_normals):
    contact = find_contact_region(outline_pixels, index_array, [r, c], ang)
    l1, l2, l3, l4, l5, va = grasp_features(contact, outline_pixels, outline_normals, [r, c], ang)
    score = get_score([l5, va], l2, l3, 2)
    data = [score, r, c, ang, l5, va, l1, l2, l3, l4]  # current grasp data
    return data


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, angle, drawing, img, img2, results, outline_pixels, outline_normals, index_array, Im
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
        angle = np.rad2deg(np.arctan2(y - y1, x - x1))
        plot_grasp_path([y1, x1], angle, 19, 130, img)
        contact_reg = extract_contact_region(y1, x1, angle, 19, 130, Im[:, :, 0])
        cv_plot_contact_region(contact_reg, img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            a, b = x, y
            if a != x & b != y:
                img = img2.copy()
                angle = np.rad2deg(np.arctan2(y - y1, x - x1))
                plot_grasp_path([y1, x1], angle, 19, 130, img)
                contact_reg = extract_contact_region(y1, x1, angle, 19, 130, Im[:, :, 0])
                cv_plot_contact_region(contact_reg, img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        angle = np.rad2deg(np.arctan2(y - y1, x - x1))
        plot_grasp_path([y1, x1], angle, 19, 130, img)
        contact_reg = extract_contact_region(y1, x1, angle, 19, 130, Im[:, :, 0])
        cv_plot_contact_region(contact_reg, img)
        data_i = calculate_data(y1, x1, angle, outline_pixels, index_array, outline_normals)
        results = np.append(results, [data_i], axis=0)
        print('data recorded', data_i)


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    img_rgb = 'wbt.png'
    img_mask = 'spoon3_mask.png'
    I = cv2.imread(os.path.join(path, 'pictures', img_rgb))
    Im = cv2.imread(os.path.join(path, 'pictures', img_mask))
    Ip = get_object_pixels(Im[:, :, 0])
    outline_pixels, outline_normals, index_array = get_outline_and_normal(Im[:, :, 0], 0, 0, 7)
    results = np.ndarray((0, 10), dtype=np.float16)

    windowName = 'Draw grasp rectangle'
    img = Im * 255
    img2 = img.copy()
    drawing = False
    x1 = -1
    y1 = -1
    angle = -1
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, draw_rectangle)
    while True:
        cv2.imshow(windowName, img)
        if cv2.waitKey(20) == ord('q'):
            break
    cv2.destroyAllWindows()
    # np.savetxt(f'gf_test_{time.time()}.txt', results, fmt='%1.4f')

    # # individual test///////////////////////////
    # ims=Im[:,:,0]*255
    # plot_grasp_path([417, 630], -1.7306, 19, 130, ims)
    # cv2.imshow('123', ims)
    # outline_pixels, outline_normals, index_array = get_outline_and_normal(Im[:, :, 0], 0, 0, 7)
    # contact = find_contact_region(outline_pixels, index_array, [417, 630], -1.7306)
    # l1, l2, l3 = grasp_features(contact, outline_pixels, outline_normals, [417, 630], -1.7306)
    # print(l1, l2, l3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # gf_data = gather_data(Im, Ip)
    # np.savetxt('grasp_feature_study.txt', gf_data, fmt='%1.4f')
