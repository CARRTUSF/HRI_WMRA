import numpy as np
import math


def normalize(n):
    norm = (n[0] ** 2 + n[1] ** 2) ** 0.5
    n[0] /= norm
    n[1] /= norm
    return n


def get_point_type(pt, mask):
    try:
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
    except:
        point_type = 0
    return point_type


def find_contact_seed_point(r_c, c_c, sin_theta, cos_theta, w, mask, search_resolution=4):
    for i in range(int(math.log(w / search_resolution, 2)) + 1):
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
            elif not left_contact_found and l2 - l1 <= 1:
                left_contact_point = [-1, -1, 1, -2 * w - 1, hn, 0, 0]  # left side collision
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


def rotation_angle(l_profile, r_profile, l_min, r_max, contact_threshold=1):
    left_angle = 0
    right_angle = 0
    n = l_profile.size
    sl1 = n
    sl2 = -1
    sr1 = n
    sr2 = -1
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
    if sr1 > sl2 and sl1 != n and sr1 != n and sl2 != -1 and sr2 != -1:
        rot_center = int((sr1 + sl2) / 2)
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
    return min(left_angle, right_angle)


def slippage_angle(l_profile, r_profile, l_normals, r_normals, theta, l_min, r_max, contact_threshold=3):
    l_contact_points_ids = np.ndarray((0,), dtype=np.int8)
    r_contact_points_ids = np.ndarray((0,), dtype=np.int8)
    left_slippage_angle, right_slippage_angle = 0, 0
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
    if abs(left_slippage_angles_sum1) == abs(left_slippage_angles_sum2) and left_contact_count != 0:
        left_slippage_angle = left_slippage_angles_sum1 / left_contact_count
    if abs(right_slippage_angles_sum1) == abs(right_slippage_angles_sum2) and right_contact_count != 0:
        right_slippage_angle = right_slippage_angles_sum1 / right_contact_count
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


def high_level_grasp_feature(left_contact_region, right_contact_region, theta, h, w):
    l_profile = left_contact_region[:, 3]
    r_profile = right_contact_region[:, 3]
    l_normals = left_contact_region[:, 5:]
    r_normals = right_contact_region[:, 5:]
    l_min = 2 * w + 2
    r_max = -2 * w + 2

    collision = False
    translation = 200.0
    rot_ang = 180.0
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


def grasp_quality_score_v2(collision, translation, rot, slip, contact_offset, gripper_hh, gripper_hw):

    if collision:
        score = 0.0
    elif translation == -1:  # no object detected
        score = -1.0
    else:
        slip_ang = (abs(slip[0]) + abs(slip[1])) / 2
        # contact_offset = contact_offsets[0] + contact_offsets[1]
        translation = translation/gripper_hw
        s1 = linearly_normalized_score(translation, 1, [0.0, 1.0, 1.0, 0.0])
        s2 = linearly_normalized_score(rot, 1, [0.0, 1.0, 60.0, 0.0])
        s3 = linearly_normalized_score(slip_ang, 1, [0.0, 1.0, 60.0, 0.0])
        s4 = linearly_normalized_score(contact_offset, 2, [0.0, 1.0, 0.5 * gripper_hh, 0.7, gripper_hh, 0.0])
        score = combine_score_v2([s1, s2, s3, s4])
    return score


def evaluate_grasp(grasp, mask):
    # grasp=[row,col,angle,hh,hw]
    contact_rl, contact_rr = extract_contact_region(grasp[0], grasp[1], grasp[2], grasp[3], grasp[4], mask)
    cli, trans, rotation, slippage, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr, grasp[2],
                                                                                  grasp[3], grasp[4])
    score = grasp_quality_score_v2(cli, trans, rotation, slippage, offs, grasp[3], grasp[4])
    return score
