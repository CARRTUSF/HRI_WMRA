import cv2
import numpy as np
import math
import os
import time
from vgt2 import find_contact_region, get_outline_and_normal


def pad_zeros(img, x):
    r, c = img.shape
    padded_img = np.zeros((r + 2 * x, c + 2 * x), np.uint8)
    padded_img[x:x + r, x:x + c] = img
    return padded_img


def l2_distance(p1, p2, option=0):
    distance = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    if option == 0:
        return distance ** 0.5
    else:
        return distance


def normalize(n):
    norm = (n[0] ** 2 + n[1] ** 2) ** 0.5
    n[0] /= norm
    n[1] /= norm
    return n


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


def get_corners(grasparam, h, w):
    """

    :param grasparam: [row,col,angle] (row,col) center of the grasp
    :param h:
    :param w:
    :return:
    """
    theta = np.deg2rad(grasparam[2])
    corner_vectors_original = np.array([[-h, h, h, -h], [-w, -w, w, w], [1, 1, 1, 1]])
    transformation = np.array(
        [[np.cos(theta), np.sin(theta), grasparam[0]], [-np.sin(theta), np.cos(theta), grasparam[1]], [0, 0, 1]])
    new_corners = np.matmul(transformation, corner_vectors_original)
    return new_corners


def plot_grasp_rect(grasparam, h, w, img, color=(0, 0, 255)):
    new_corners = get_corners(grasparam, h, w)
    for k in range(4):
        if k != 3:
            cv2.line(img, (int(round(new_corners[1][k])), int(round(new_corners[0][k]))),
                     (int(round(new_corners[1][k + 1])), int(round(new_corners[0][k + 1]))), color)
        else:
            cv2.line(img, (int(round(new_corners[1][3])), int(round(new_corners[0][3]))),
                     (int(round(new_corners[1][0])), int(round(new_corners[0][0]))), color)
    cv2.circle(img, (int(grasparam[1]), int(grasparam[0])), 3, color, -1)


def plot_contact_profile(profile, img, side, x, y, theta, w, r, contact_ids=None):
    space = 10
    cos_0 = np.cos(np.deg2rad(theta))
    sin_0 = np.sin(np.deg2rad(theta))
    n = profile.shape[0]
    if side == 'left':
        dr_j1 = -r / 2 - space
        c_offset = 2
    else:
        dr_j1 = r / 2 + space
        c_offset = -2
    if contact_ids is None:
        contact_ids = []
    for wi in range(n):
        # if side != 'left':
        #     wi = n-wi-1
        dw_i = -w*(2*wi-n-1)/(2*(n-1))
        h = profile[wi]
        dr_j2 = dr_j1 + h
        ri = int(round(y + dw_i*cos_0 + dr_j1*sin_0))
        ci = int(round(x - dw_i*sin_0 + dr_j1*cos_0))
        ri_e = int(round(y + dw_i*cos_0 + dr_j2*sin_0))
        ci_e = int(round(x - dw_i*sin_0 + dr_j2*cos_0))
        color = (166, 247, 247)
        cv2.line(img, (ci, ri), (ci_e, ri_e), color, 2)
    # color = (0, 0, 255)
    for cid in contact_ids:
        # if side != 'left':
        #     cid = n - cid - 1
        dw_i = -w*(2*cid-n-1)/(2*(n-1))
        dr_j3 = dr_j1 + profile[cid] + c_offset
        ri_c = int(round(y + dw_i*cos_0 + dr_j3*sin_0))
        ci_c = int(round(x - dw_i*sin_0 + dr_j3*cos_0))
        cv2.circle(img, (ci_c, ri_c), 1, (0, 0, 255), -1)


def cv_plot_contact_region(contact_region, img, plot_normal=False, contact_ids=np.array([]), color1=(200, 200, 0),
                           color2=(200, 0, 200)):
    if contact_region.shape[0] != 0:
        if contact_region.shape[1] > 2:
            for ct in contact_region:
                if ct[0] != -1:
                    if ct[2] == 1:  # left/up contact
                        cv2.circle(img, (int(ct[1]), int(ct[0])), 2, color1, -1)
                    elif ct[2] == 2:  # right/down contact
                        cv2.circle(img, (int(ct[1]), int(ct[0])), 2, color2, -1)
                    if plot_normal and contact_ids.size == 0:
                        end_p = (int(ct[1] + 7 * ct[-1]), int(ct[0] + 7 * ct[-2]))
                        cv2.line(img, (int(ct[1]), int(ct[0])), end_p, (0, 255, 0))
            if plot_normal and contact_ids.size != 0:
                for i in contact_ids:
                    ct = contact_region[i]
                    end_p = (int(ct[1] + 10 * ct[-1]), int(ct[0] + 10 * ct[-2]))
                    cv2.line(img, (int(ct[1]), int(ct[0])), end_p, (0, 255, 0))
        elif contact_region.shape[1] == 2:
            for ct in contact_region:
                cv2.circle(img, (int(ct[1]), int(ct[0])), 2, color1)


def get_grasp_rect_corners(center, theta, h, w):
    """

    :param center:
    :param theta:
    :param h:
    :param w:
    :return: [
    """
    theta = np.deg2rad(theta)
    corner_vectors_original = np.array([[-h, h, h, -h], [-w, -w, w, w], [1, 1, 1, 1]])
    transformation = np.array(
        [[np.cos(theta), np.sin(theta), center[0]], [-np.sin(theta), np.cos(theta), center[1]], [0, 0, 1]])
    new_corners = np.matmul(transformation, corner_vectors_original)
    return new_corners


def get_point_type(pt, mask):
    # print('p1', mask[pt[0]-1, pt[1]-1], mask[pt[0]-1, pt[1]], mask[pt[0]-1, pt[1]+1])
    # print('p2', mask[pt[0], pt[1] - 1], mask[pt[0], pt[1]], mask[pt[0], pt[1] + 1])
    # print('p3', mask[pt[0] + 1, pt[1] - 1], mask[pt[0] + 1, pt[1]], mask[pt[0] + 1, pt[1] + 1])
    if mask[pt[0], pt[1]] == 1:
        boundary_detector = mask[pt[0] - 1, pt[1] - 1] * mask[pt[0] - 1, pt[1]] * mask[pt[0] - 1, pt[1] + 1] * \
                            mask[pt[0], pt[1] - 1] * mask[pt[0], pt[1]] * mask[pt[0], pt[1] + 1] * \
                            mask[pt[0] + 1, pt[1] - 1] * mask[pt[0] + 1, pt[1]] * mask[pt[0] + 1, pt[1] + 1]
        if boundary_detector == 0:
            point_type = 2  # boundary point
        else:
            point_type = 1  # object point / inside point
    else:
        point_type = 0  # not object point / outside point
    return point_type


def get_boundary_pixel_normal(boundary_pixel, mask, x, neighbor_vectors_row, neighbor_vectors_col):
    r0 = boundary_pixel[0] - x
    c0 = boundary_pixel[1] - x
    window_weights = [[0, 0.314, 0.813, 1, 0.813, 0.314, 0],
                      [0.314, 1, 1, 1, 1, 1, 0.314],
                      [0.813, 1, 1, 1, 1, 1, 0.813],
                      [1, 1, 1, 1, 1, 1, 1],
                      [0.813, 1, 1, 1, 1, 1, 0.813],
                      [0.314, 1, 1, 1, 1, 1, 0.314],
                      [0, 0.314, 0.813, 1, 0.813, 0.314, 0]]
    window_weights = np.array(window_weights)  # change the window shape to a circle of radius x
    window_mask = mask[r0:r0 + 2 * x + 1, c0:c0 + 2 * x + 1]
    occupied_neighbor_vectors_row = neighbor_vectors_row * window_mask * window_weights
    occupied_neighbor_vectors_col = neighbor_vectors_col * window_mask * window_weights
    n_r = np.sum(occupied_neighbor_vectors_row)
    n_c = np.sum(occupied_neighbor_vectors_col)
    neighbor_vectors_sum = normalize([n_r, n_c])
    normal = [-neighbor_vectors_sum[0], -neighbor_vectors_sum[1]]
    return normal


def find_contact_seed_point(r_c, c_c, sin_theta, cos_theta, w, mask, search_resolution=4):
    for i in range(int(math.log2(w / search_resolution)) + 1):
        for j in range(int((2 ** i - 1) / 2) + 1):
            for sign in range(2):
                s = int((2 * sign - 1) * 2 ** (-i) * (2 * j + 1) * w)
                dsr = s * sin_theta
                dsc = s * cos_theta
                rt = int(round(r_c + dsr))
                ct = int(round(c_c + dsc))
                point_type = get_point_type([rt, ct], mask)
                if point_type == 1:
                    return s
    # print('no contact seed :(')
    return -1


def gpl_intersection_points(r0, c0, sin_theta, cos_theta, hn, w, mask, window_size):
    # -------------------calculate normals----------------------------
    x = int(window_size / 2)
    v = np.arange(-x, x + 1)
    neighbor_vectors_row = np.repeat(v.reshape((-1, 1)), window_size, axis=1)
    neighbor_vectors_col = np.repeat(v.reshape((1, -1)), window_size, axis=0)
    # ----------------------------------------------------------------------
    intsec_points = np.ndarray((0, 7), dtype=np.float16)
    dcr = hn * cos_theta
    dcc = hn * sin_theta
    r_c = r0 - dcr
    c_c = c0 + dcc
    # start_d = [0, -65, 65, -33, 33, 98, -98, -16, 16, 49, -49, -82, 82, -114, 114, -8, 8, -24, 24, -41, 41, -57, 57,
    #            -74, 74, -90, 90, -106, 106, -122, 122, -4, 4, -12, 12, -20, 20, -28, 28, -37, 37, -45, 45, -53, 53,
    #            -61, 61, -70, 70, -78, 78, -86, 86, -94, 94, -102, 102, -110, 110, -118, 118, -126, 126]
    # start_d = generate_search_index2(w, 4)
    # for s in start_d:
    #     dsr = s * sin_theta
    #     dsc = s * cos_theta
    #     if s_found == -1:
    #         rt = int(round(r_c + dsr))
    #         ct = int(round(c_c + dsc))
    #         point_type = get_point_type([rt, ct], mask)
    #         if point_type == 1:
    #             s_found = s
    #     else:
    #         break

    # find seed point
    s_found = find_contact_seed_point(r_c, c_c, sin_theta, cos_theta, w, mask, search_resolution=2)
    if s_found != -1:  # the grasp line intersects the object mask
        l1 = -w
        l2 = s_found
        r1 = s_found
        r2 = w
        left_contact_found = False
        right_contact_found = False
        while not (left_contact_found and right_contact_found):
            if not left_contact_found and l2 - l1 > 1:
                lm = (l1 + l2) / 2
                # calculate left test point
                lrt = int(round(r_c + lm * sin_theta))
                lct = int(round(c_c + lm * cos_theta))
                point_type_l = get_point_type([lrt, lct], mask)
                if point_type_l == 2:
                    normal1 = get_boundary_pixel_normal([lrt, lct], mask, x, neighbor_vectors_row, neighbor_vectors_col)
                    left_contact_point = [lrt, lct, 1, lm, hn, normal1[0], normal1[1]]
                    intsec_points = np.append(intsec_points, [left_contact_point], axis=0)
                    left_contact_found = True
                elif point_type_l == 0:
                    l1 = lm
                else:
                    l2 = lm
<<<<<<< HEAD
            elif not left_contact_found and l2 - l1 < 1:
                left_contact_point = [-1, -1, 1, -2 * w - 1, hn, 0, 0]  # left side collision
=======
            elif not left_contact_found and l2 - l1 <= 1:
                left_contact_point = [-1, -1, 1, -2 * w - 1, hn, 0, 0]  # left side collision
>>>>>>> eb3785daeb2217ec6f56eb195fa2d9c070f20410
                intsec_points = np.append(intsec_points, [left_contact_point], axis=0)
                left_contact_found = True

            if not right_contact_found and r2 - r1 > 1:
                rm = (r1 + r2) / 2
                # calculate right test point
                rrt = int(round(r_c + rm * sin_theta))
                rct = int(round(c_c + rm * cos_theta))
                point_type_r = get_point_type([rrt, rct], mask)
                if point_type_r == 2:
                    normal2 = get_boundary_pixel_normal([rrt, rct], mask, x, neighbor_vectors_row, neighbor_vectors_col)
                    right_contact_point = [rrt, rct, 2, rm, hn, normal2[0], normal2[1]]
                    intsec_points = np.append(intsec_points, [right_contact_point], axis=0)
                    right_contact_found = True
                elif point_type_r == 0:
                    r2 = rm
                else:
                    r1 = rm
            elif not right_contact_found and r2 - r1 <= 1:
                right_contact_point = [-1, -1, 2, 2 * w + 1, hn, 0, 0]  # right side collision
                intsec_points = np.append(intsec_points, [right_contact_point], axis=0)
                right_contact_found = True
    else:  # the grasp line does not intersect the object mask
        left_contact_point = [-1, -1, 1, -2 * w, hn, 0, 0]  # no contact
        intsec_points = np.append(intsec_points, [left_contact_point], axis=0)
        right_contact_point = [-1, -1, 2, 2 * w, hn, 0, 0]  # no contact
        intsec_points = np.append(intsec_points, [right_contact_point], axis=0)
    return intsec_points


def extract_contact_region(r0, c0, theta, h, w, mask):
    left_contact_region = np.ndarray((0, 7), dtype=np.float16)
    right_contact_region = np.ndarray((0, 7), dtype=np.float16)
    sin_theta = np.sin(np.deg2rad(theta))
    cos_theta = np.cos(np.deg2rad(theta))
    for hn in range(-h, h + 1):
        intersection_pts = gpl_intersection_points(r0, c0, sin_theta, cos_theta, hn, w, mask, 7)
        if intersection_pts.shape[0] == 2:
            if intersection_pts[0][2] == 1:
                left_contact_region = np.append(left_contact_region, [intersection_pts[0]], axis=0)
                right_contact_region = np.append(right_contact_region, [intersection_pts[1]], axis=0)
            else:
                left_contact_region = np.append(left_contact_region, [intersection_pts[1]], axis=0)
                right_contact_region = np.append(right_contact_region, [intersection_pts[0]], axis=0)
    return left_contact_region, right_contact_region


def extract_contact_points(contact_region, threshold=8):
    left_contacts = np.ndarray((0, 7), dtype=np.float16)
    right_contacts = np.ndarray((0, 7), dtype=np.float16)
    contact_points = np.ndarray((0, 7), dtype=np.float16)
    min_dl = 1500
    max_dr = -1500
    nl = 0
    nr = 0
    if contact_region.shape[0] != 0:
        for cp in contact_region:
            if cp[2] == 1:
                left_contacts = np.append(left_contacts, [cp], axis=0)
                if cp[3] < min_dl:
                    min_dl = cp[3]
            else:
                right_contacts = np.append(right_contacts, [cp], axis=0)
                if cp[3] > max_dr:
                    max_dr = cp[3]
        for lp in left_contacts:
            if abs(lp[3] - min_dl) < threshold:
                nl += 1
                contact_points = np.append(contact_points, [lp], axis=0)
        for rp in right_contacts:
            if abs(rp[3] - max_dr) < threshold:
                nr += 1
                contact_points = np.append(contact_points, [rp], axis=0)
    return contact_points, nl, nr


def rotation_angle(l_profile, r_profile, l_min, r_max, contact_threshold=1):
    left_angle = 0
    right_angle = 0
    n = l_profile.size
    sl1 = n
    sl2 = -1
    sr1 = n
    sr2 = -1
    # rot_center = -1
    # rot_direction = 0
    for i in range(n):
        if abs(l_profile[i] - l_min) < contact_threshold:
            if i < sl1:
                sl1 = i
            if i > sl2:
                sl2 = i
        if abs(r_profile[i] - r_max) < contact_threshold:
            if i < sr1:
                sr1 = i
            if i > sr2:
                sr2 = i

    # rotation angles-------------------------
    if sl1 > sr2 and sl1 != n and sl2 != -1 and sr1 != n and sr2 != -1:  # left up right down
        rot_center = int((sl1 + sr2) / 2)
        # rot_direction = -1
        for i in range(rot_center):
            dh = abs(l_profile[sl1] - l_profile[i])
            ds = abs(i - sl1)
            if ds == 0:
                ang_i = 0
            else:
                tan_i = dh / ds
                ang_i = np.float16(np.rad2deg(np.arctan(tan_i)))
            if i == 0:
                left_angle = ang_i
            elif ang_i < left_angle:
                left_angle = ang_i
            # left_angles = np.append(left_angles, [ang_i], axis=0)
        for j in range(rot_center, n):
            dh = abs(r_profile[sr2] - r_profile[j])
            ds = abs(j - sr2)
            if ds == 0:
                ang_i = 0
            else:
                tan_i = dh / ds
                ang_i = np.float16(np.rad2deg(np.arctan(tan_i)))
            if j == rot_center:
                right_angle = ang_i
            elif ang_i < right_angle:
                right_angle = ang_i
            # right_angles = np.append(right_angles, [ang_i], axis=0)
    if sr1 > sl2 and sl1 != n and sr1 != n and sl2 != -1 and sr2 != -1:
        rot_center = int((sr1 + sl2) / 2)
        # rot_direction = 1
        for i in range(rot_center, n):
            dh = abs(l_profile[sl2] - l_profile[i])
            ds = abs(i - sl2)
            if ds == 0:
                ang_i = 0
            else:
                tan_i = dh / ds
                ang_i = np.float16(np.rad2deg(np.arctan(tan_i)))
            if i == rot_center:
                left_angle = ang_i
            elif ang_i < left_angle:
                left_angle = ang_i
            # left_angles = np.append(left_angles, [ang_i], axis=0)
        for j in range(rot_center):
            dh = abs(r_profile[sr1] - r_profile[j])
            ds = abs(j - sr1)
            if ds == 0:
                ang_i = 0
            else:
                tan_i = dh / ds
                ang_i = np.float16(np.rad2deg(np.arctan(tan_i)))
            if j == 0:
                right_angle = ang_i
            elif ang_i < right_angle:
                right_angle = ang_i
    # print('rotation angles and center:', left_angle, right_angle, rot_center)
    return min(left_angle, right_angle)


def slippage_angle(l_profile, r_profile, l_normals, r_normals, theta, l_min, r_max, contact_threshold=3):
    l_contact_points_ids = np.ndarray((0,), dtype=np.int8)
    r_contact_points_ids = np.ndarray((0,), dtype=np.int8)
    left_slippage_angle, right_slippage_angle = 0, 0
    # left_slippage_angle2, right_slippage_angle2 = 0, 0  # for comparision
    sin_theta = np.sin(np.deg2rad(theta))
    cos_theta = np.cos(np.deg2rad(theta))
    grasp_direction = [sin_theta, cos_theta]
    left_slippage_angles = np.ndarray((0,), dtype=np.float16)
    right_slippage_angles = np.ndarray((0,), dtype=np.float16)
    left_slippage_angles_sum1 = 0
    left_slippage_angles_sum2 = 0
    right_slippage_angles_sum1 = 0
    right_slippage_angles_sum2 = 0
    left_contact_count = 0
    right_contact_count = 0
    rot_m = [[cos_theta, -sin_theta],
             [sin_theta, cos_theta]]  # rotation from image coordinate system to gripper coordinate system
    for i in range(l_profile.size):
        if abs(l_profile[i] - l_min) < contact_threshold:  # test contact points
            l_normal_g = np.matmul(rot_m, l_normals[i])
            dcl = np.float16(np.dot(grasp_direction, l_normals[i]))
            left_slippage_angle_i = 180 - np.rad2deg(np.arccos(dcl))
            left_slippage_angles_sum1 += left_slippage_angle_i
            if l_normal_g[0] < 0:
                left_slippage_angles = np.append(left_slippage_angles, [-left_slippage_angle_i], axis=0)
                left_slippage_angles_sum2 -= left_slippage_angle_i
            else:
                left_slippage_angles = np.append(left_slippage_angles, [left_slippage_angle_i], axis=0)
                left_slippage_angles_sum2 += left_slippage_angle_i
            l_contact_points_ids = np.append(l_contact_points_ids, [i], axis=0)
            left_contact_count += 1
        if abs(r_profile[i] - r_max) < contact_threshold:
            r_normal_g = np.matmul(rot_m, r_normals[i])
            dcr = np.float16(np.dot(grasp_direction, r_normals[i]))
            right_slippage_angle_i = np.rad2deg(np.arccos(dcr))
            right_slippage_angles_sum1 += right_slippage_angle_i
            if r_normal_g[0] < 0:
                right_slippage_angles = np.append(right_slippage_angles, [-right_slippage_angle_i], axis=0)
                right_slippage_angles_sum2 -= right_slippage_angle_i
            else:
                right_slippage_angles = np.append(right_slippage_angles, [right_slippage_angle_i], axis=0)
                right_slippage_angles_sum2 += right_slippage_angle_i
            r_contact_points_ids = np.append(r_contact_points_ids, [i], axis=0)
            right_contact_count += 1
    # left_contact_normal = left_contact_normal_sum / left_contact_count
    # right_contact_normal = right_contact_normal_sum / right_contact_count
    # left_slippage_angle = 180 - np.rad2deg(np.arccos(np.dot(grasp_direction, left_contact_normal)))
    # right_slippage_angle = np.rad2deg(np.arccos(np.dot(grasp_direction, right_contact_normal)))
    if abs(left_slippage_angles_sum1) == abs(left_slippage_angles_sum2) and left_contact_count != 0:
        # print('left will slip**************', left_slippage_angles_sum1/left_contact_count)
        # left_slippage_angles_abs = np.abs(left_slippage_angles)
        # ls_max = np.amax(left_slippage_angles_abs)
        # ls_min = np.amin(left_slippage_angles_abs)
        # left_slippage_angle2 = (left_slippage_angles_sum1 - ls_max - ls_min)/(left_contact_count - 2)
        left_slippage_angle = left_slippage_angles_sum1 / left_contact_count
    if abs(right_slippage_angles_sum1) == abs(right_slippage_angles_sum2) and right_contact_count != 0:
        # print('right will slip*************', right_slippage_angles_sum1/right_contact_count)
        # right_slippage_angles_abs = np.abs(right_slippage_angles)
        # rs_max = np.amax(right_slippage_angles_abs)
        # rs_min = np.amin(right_slippage_angles_abs)
        # right_slippage_angle2 = (right_slippage_angles_sum1 - rs_max - rs_min)/(right_contact_count - 2)
        right_slippage_angle = right_slippage_angles_sum1 / right_contact_count
    # print('left slippage angles:', left_slippage_angle, left_slippage_angle2)
    # print('right slippage angles:', right_slippage_angle, right_slippage_angle2)
    return left_slippage_angle, right_slippage_angle, l_contact_points_ids, r_contact_points_ids


def contact_center_offset(l_contacts, r_contacts, gipper_hh):
    if l_contacts.size != 0:
        # print('left contact ids', np.amax(l_contacts), np.amin(l_contacts))
        lc_off = (np.amax(l_contacts) + np.amin(l_contacts)) / 2 - gipper_hh
    else:
        lc_off = gipper_hh
    if r_contacts.size != 0:
        rc_off = (np.amax(r_contacts) + np.amin(r_contacts)) / 2 - gipper_hh
        # print('right contact ids', np.amax(r_contacts), np.amin(r_contacts))
    else:
        rc_off = gipper_hh
    return abs((lc_off + rc_off) / 2)


def contact_center_offset_v2(l_contacts, r_contacts, gripper_hh):
    if l_contacts.size != 0:
        # print('left contact ids', np.amax(l_contacts), np.amin(l_contacts))
        lc = (np.amax(l_contacts) + np.amin(l_contacts)) / 2
    else:
        lc = gripper_hh
    if r_contacts.size != 0:
        rc = (np.amax(r_contacts) + np.amin(r_contacts)) / 2
        # print('right contact ids', np.amax(r_contacts), np.amin(r_contacts))
    else:
        rc = gripper_hh
    f1 = abs((lc + rc) / 2 - gripper_hh)
    f2 = abs(lc - rc)
    return [f1, f2]


def high_level_grasp_feature(left_contact_region, right_contact_region, theta, h, w):
    # contact_center_offset(Y), translation(x), rotation(roll), collision, slippage
    l_profile = left_contact_region[:, 3]
    r_profile = right_contact_region[:, 3]
    l_normals = left_contact_region[:, 5:]
    r_normals = right_contact_region[:, 5:]
    l_min = 2 * w + 2
    r_max = -2 * w + 2

    collision = False
    translation = 200.0
    rot_ang = 180.0
    # rot_dir = 0
    l_slip_ang = 180.0
    r_slip_ang = 180.0
    gripper_offset = h
    lcids = np.ndarray((0,), dtype=np.int8)
    rcids = np.ndarray((0,), dtype=np.int8)
    # -------find primary contact point and check for collision
    for pt in l_profile:
        if pt == -2 * w - 1:
            collision = True
            break
        elif -w <= pt < l_min:
            l_min = pt
    for pt in r_profile:
        if pt == 2 * w + 1:
            collision = True
            break
        elif w >= pt > r_max:
            r_max = pt
    # ---------------------------------------------------------
    if not collision:
        if np.amax(l_profile) == -2 * w:
            # print('no object detected$$$$$$$$$$$$$$')
            translation = -1
        else:
            # --------------------------translation
            translation = abs((l_min + r_max) / 2.0)
            # --------------------------rotation
            rot_ang = rotation_angle(l_profile, r_profile, l_min, r_max)
            l_slip_ang, r_slip_ang, lcids, rcids = slippage_angle(l_profile, r_profile, l_normals, r_normals, theta,
                                                                  l_min, r_max, 5)
            gripper_offset = contact_center_offset(lcids, rcids, h)
    return collision, translation, rot_ang, [l_slip_ang, r_slip_ang], lcids, rcids, gripper_offset


def linearly_normalized_score(feature, n, kernel_profile, feature_uncertainty=-1):
    score = 0.0
    score_uncertainty = 0.0
    for i in range(n):
        xi = kernel_profile[2 * (i + 1)]
        if feature <= xi:
            x0 = kernel_profile[2 * i]
            y0 = kernel_profile[2 * i + 1]
            yi = kernel_profile[2 * (i + 1) + 1]
            k = (yi - y0) / (xi - x0)
            score = k * feature + y0 - k * x0
            score_uncertainty = abs(k) * feature_uncertainty
            break
    if feature_uncertainty == -1:
        return score
    else:
        return [score, score_uncertainty]


def combine_scores(scores):
    features = np.array(scores)[:, 0]
    us = np.array(scores)[:, 1]
    feature_sum = np.sum(features)
    if feature_sum == 0:
        weights = np.repeat(1 / features.shape[0], features.shape[0])
    else:
        init_weights = features / feature_sum
        invert_init_weights = 1 - init_weights
        weights = invert_init_weights / np.sum(invert_init_weights)
    # print('-------weights assignment-------')
    # print(features)
    # print(weights)
    score = np.dot(weights, features)
    squared_sum_uncertainties = 0
    for i in range(us.size):
        squared_sum_uncertainties += (weights[i] * us[i]) ** 2
    score_uncertainty = np.sqrt(squared_sum_uncertainties)
    # score = (1-s_min_weight)/3 * (s1 + s2 + s3 + s4) + (4*s_min_weight-1)/3 * s_min[0]
    # print('score test', s1, s2, s3, s4, score, score_uncertainty)
    # print('params', s_min, s_min_weight)
    return [score, score_uncertainty]


def grasp_quality_score(collision, translation, rot, slip, contact_offset, gripper_hh):
    s1 = [0.0, 0.0]
    s2 = [0.0, 0.0]
    s3 = [0.0, 0.0]
    s4 = [0.0, 0.0]
    if collision:
        score = [0.0, 0.0]
    elif translation == -1:
        score = [-1.0, 0.0]
    else:
        slip_ang = (abs(slip[0]) + abs(slip[1])) / 2
        # contact_offset = contact_offsets[0] + contact_offsets[1]
        s1 = linearly_normalized_score(translation, 1, [0, 1, 100, 0], 3)
        s2 = linearly_normalized_score(rot, 1, [0, 1, 70, 0], 4)
        s3 = linearly_normalized_score(slip_ang, 1, [0, 1, 70, 0], 10)
        s4 = linearly_normalized_score(contact_offset, 2, [0, 1, 0.75 * gripper_hh, 0.7, gripper_hh, 0], 1)
        score = combine_scores([s1, s2, s3, s4])
    return [score, s1, s2, s3, s4]


def grasp_quality_score_v2(collision, translation, rot, slip, contact_offset, gripper_hh, gripper_hw, new=False):
    # s1 = 0.0
    # s2 = 0.0
    # s3 = 0.0
    # s4 = 0.0
    if collision:
        score = 0.0
    elif translation == -1:  # no object detected
        score = -1.0
    else:
        slip_ang = (abs(slip[0]) + abs(slip[1])) / 2.0
        if new:
            translation = translation/gripper_hw
            s1 = linearly_normalized_score(translation, 1, [0.0, 1.0, 1.0, 0.0])
        else:
            s1 = linearly_normalized_score(translation, 1, [0, 1, 100, 0])
        # contact_offset = contact_offsets[0] + contact_offsets[1]
        s2 = linearly_normalized_score(rot, 1, [0, 1, 60, 0])
        s3 = linearly_normalized_score(slip_ang, 1, [0, 1, 60, 0])
        s4 = linearly_normalized_score(contact_offset, 2, [0, 1, 0.5 * gripper_hh, 0.7, gripper_hh, 0])
        score = combine_score_v2([s1, s2, s3, s4])
    return score


def generate_search_index2(w, resolution):
    search_index = [0]
    for i in range(int(math.log2(w / resolution)) + 1):
        for j in range(int((2 ** i - 1) / 2) + 1):
            for sign in range(2):
                s = int((2 * sign - 1) * 2 ** (-i) * (2 * j + 1) * w)
                # if s == 75:
                #     print(search_index)
                #     return -1
                search_index.append(s)
    return search_index


def combine_score_v2(scores):
    scores = np.array(scores).reshape((1, -1))
    s_min = np.amin(scores)
    # print(scores)
    s_min_str = list(str(format(s_min, 'f')))
    # print(s_min_str)
    while len(s_min_str) < 3:
        s_min_str.append('0')
    x_str = '0.0'
    for ci in range(len(s_min_str)):
        if s_min_str[ci] != '0' and s_min_str[ci] != '.':
            x_str = ''.join(s_min_str[:ci + 1])
            break
    x = float(x_str)
    dx_str = list(x_str)
    dx_str[-1] = '1'
    dx = float(''.join(dx_str))
    ds_sum = 0
    for score in scores[0, :]:
        ds_sum += score - x
    p = dx * ds_sum / (3 * (1 - x) + dx)
    final_score = x + p
    # print(final_score)
    return final_score


# def combine_scores_current(scores):
#     features = np.array(scores)[0, :]
#     feature_sum = np.sum(features)
#     if feature_sum == 0:
#         weights = np.repeat(1 / features.shape[0], features.shape[0])
#     else:
#         init_weights = features / feature_sum
#         invert_init_weights = 1 - init_weights
#         weights = invert_init_weights / np.sum(invert_init_weights)
#     score = np.dot(weights, features)
#     return score


if __name__ == '__main__':
    # si2 = generate_search_index2(100, 1)
    # print(si2)
    # print(len(si2))

    for i in range(20000):
        scores_test = np.random.rand(1, 4)
        sf = combine_score_v2(scores_test)
        print(scores_test)
        print(sf)
        if sf > 1:
            print('$$$$$$$$$$$$$')
            break
        print('***************')
    # print('----------------')
    # # st = [0, 0.8, 0.7, 0.9]
    # st = [9.644e-01, 6.982e-01, 2.537e-01, 6.431e-01]
    # sf = combine_score_v2(st)
    # print(sf)
    # features = np.array([0.9, 1, 0.5, 0.05])
    # feature_sum = np.sum(features)
    # init_weights = features / feature_sum
    # invert_init_weights = 1 - init_weights
    # weights = invert_init_weights / np.sum(invert_init_weights)
    #
    # print(features)
    # print(feature_sum)
    # print('initial weights: ', init_weights)
    # print(invert_init_weights)
    # print(weights)
    # print(np.dot(weights, features))
    #
    # invert_features = 1 - features
    # invert_features_sum = np.sum(invert_features)
    # init_weights = invert_features / invert_features_sum
    # invert_init_weights = 1 - init_weights
    # weights = invert_init_weights / np.sum(invert_init_weights)
    #
    # print(weights)
    # print(np.dot(weights, features))

    # path = os.path.dirname(os.getcwd())
    # img_rgb = 'wbt.png'
    # img_mask = 'wbt_mask.png'
    # I = cv2.imread(os.path.join(path, 'pictures', img_rgb))
    # Im = cv2.imread(os.path.join(path, 'pictures', img_mask))
    # outline_pixels, outline_normals, index_array = get_outline_and_normal(Im[:, :, 0], 0, 0, 7)
    # time.sleep(1)
    # start_time = time.time()
    # # contact_rg, centering_score, ns, nd, x5, x6 = extract_contact_region(400, 617, 25.6109, 19, 130, Im[:, :, 0])
    # # contact_rg = v3_contact_region(400, 617, 25.6109, 19, 130, Im[:, :, 0])
    # # contact_rg = get_contact_region(393, 621, 71, outline_pixels, index_array)
    # print(f'new method find contact region used {time.time() - start_time} seconds')
    #
    # # time.sleep(1)
    # # start_time = time.time()
    # # contact = find_contact_region(outline_pixels, index_array, [393, 621], -5)
    # # print(f'old method find contact region used {time.time() - start_time} seconds')
    #
    # # contact_pts, n1, n2 = extract_contact_points(contact_rg)
    # # print(contact_pts)
    # # t_s = grasp_torque_feature(contact_pts, n1, n2)
    # # print('centering score', centering_score)
    # # print('contact wrench sum', t_s)
    # img_s = np.copy(Im) * 255
    # # cv_plot_contact_region(contact_rg, img_s, plot_normal=False)
    # # cv_plot_contact_region(contact_pts, img_s, True, (0, 250, 150), (0, 100, 180))
    # # for p in glp_v1:
    # #     cv2.circle(img_s, (int(round(p[1])), int(round(p[0]))), 1, (100, 200, 150), -1)
    # plot_grasp_rect([400, 617, 25.6109], 19, 130, img_s)
    # cv2.imshow('test', img_s)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# -----------------------------------------------------------
#   fix slippage angle estimation
#   fix rotation angle estimation
#   fix force center estimation
#   fix score estimation
