import numpy as np
import time
import cv2


def find_contact_points(r0, c0, theta, n, w, mask):
    # points_on_gpl = np.ndarray((0, 3), dtype=np.int)
    contact_candidates = np.ndarray((0, 5), dtype=np.float16)
    contact_points = np.ndarray((0, 5), dtype=np.float16)
    # phase 1 find all pixels on the grasp path line(gpl)//////////////////////////////////////////
    pi = -1
    p1t = time.time()
    if abs(theta) == 90:
        # find grasp line key point
        r_c = r0
        c_c = c0 + n
        # find all pixels on the grasp line
        w_limit = w
        for r in range(-w_limit, w_limit + 1):
            ci = int(c_c)
            ri = int(round(r_c + r * theta / 90))  # theta=90 -w->w, theta=-90 w->-w
            pi += 1
            # points_on_gpl = np.append(points_on_gpl, [[ri, ci, pi]], axis=0)
            if is_boundary_point([ri, ci, pi], mask):
                contact_candidates = np.append(contact_candidates, [[ri, ci, pi, 0, 0]], axis=0)
    else:
        tan_theta = np.tan(np.deg2rad(theta))
        dc = (1 + tan_theta ** 2) ** 0.5
        # find grasp line key point
        r_c = r0 - n / dc
        c_c = c0 + n * tan_theta / dc
        # find all pixels on the grasp line
        w_limit = int(w * np.cos(np.deg2rad(theta)))
        ptt = time.time()
        for c in range(-w_limit, w_limit + 1):
            ci = int(c_c) + c
            ri = gpl(r0, c0, theta, n, ci)
            pi += 1
            # points_on_gpl = np.append(points_on_gpl, [[ri, ci, pi]], axis=0)
            if is_boundary_point([ri, ci, pi], mask):
                contact_candidates = np.append(contact_candidates, [[ri, ci, pi, 0, 0]], axis=0)
        print(f"phase 1 loop used {time.time() - ptt} seconds")
    print(f"phase 1 used {time.time() - p1t} seconds")
    # # phase 2 find all boundary points in all points on the gpl//////////////////////////////////////////////
    #
    # p1t = time.time()
    # for point in points_on_gpl:
    #     if is_boundary_point(point, mask):
    #         contact_candidates = np.append(contact_candidates, [[point[0], point[1], point[2], 0, 0]], axis=0)
    #
    # print(f"phase 2 used {time.time() - p1t} seconds")
    # phase 3 extract contact points, determine contact side and contact-center offset
    p1t = time.time()
    if contact_candidates.shape[0] >= 2:
        # normal contact condition
        side_indicators = contact_candidates[:, 2]
        left_id = np.where(side_indicators == np.amin(side_indicators))[0][0]
        right_id = np.where(side_indicators == np.amax(side_indicators))[0][0]
        left_contact = [contact_candidates[left_id, 0], contact_candidates[left_id, 1]]
        right_contact = [contact_candidates[right_id, 0], contact_candidates[right_id, 1]]
        left_offset = l2_distance(left_contact, [r_c, c_c])
        right_offset = l2_distance(right_contact, [r_c, c_c])
        contact_points = np.append(contact_points, [[left_contact[0], left_contact[1], 1, left_offset, n]], axis=0)
        contact_points = np.append(contact_points, [[right_contact[0], right_contact[1], 2, right_offset, n]], axis=0)
    else:
        try:
            side_indicator = contact_candidates[0, 2]
            offset = l2_distance([contact_candidates[0, 0], contact_candidates[0, 1]], [r_c, c_c])
            if side_indicator >= w_limit:
                contact_points = np.append(contact_points,
                                           [[contact_candidates[0, 0], contact_candidates[0, 1], 2, offset, n]], axis=0)
            else:
                contact_points = np.append(contact_points,
                                           [[contact_candidates[0, 0], contact_candidates[0, 1], 1, offset, n]], axis=0)
        except IndexError:
            print('no contact point--------------------------')
    print(f"phase 3 used {time.time() - p1t} seconds")
    return contact_points


# ------------------------------------------------------
def gpl(r0, c0, theta, n, column):
    """
    grasp path line function, find the point on the line given point column
    :param r0:
    :param c0:
    :param theta:
    :param n:
    :param column:
    :return:
    """
    if abs(theta) == 90:
        return -1
    else:
        tan_theta = np.tan(np.deg2rad(theta))
        dc = (1 + tan_theta ** 2) ** 0.5
        row = r0 - n / dc + tan_theta * (column - c0 - n * tan_theta / dc)
        return int(round(row))


def get_contact_points(r0, c0, theta, n, w, mask):
    contact_points = np.ndarray((0, 5), dtype=np.float16)
    # special case where theta == +-90----------------------------------------------------
    if abs(theta) == 90:
        # find grasp line key point
        r_c = r0
        c_c = c0 + n
        # phase 1 find up side contact point//////////////////////////////////////////
        stop_point = -w
        for ru in range(-w, w + 1):
            ci = int(c_c)
            ri = int(round(r_c + ru * theta / 90))  # theta=90 -w->w, theta=-90 w->-w
            if is_boundary_point([ri, ci], mask):
                offset = l2_distance([ri, ci], [r_c, c_c])
                contact_points = np.append(contact_points, [[ri, ci, 1, offset, n]], axis=0)
                stop_point = ru
                break
        # phase 2 find down side contact point//////////////////////////////////////////
        for rd in range(w, stop_point, -1):
            ci = int(c_c)
            ri = int(round(r_c + rd * theta / 90))  # theta=90 -w->w, theta=-90 w->-w
            if is_boundary_point([ri, ci], mask):
                offset = l2_distance([ri, ci], [r_c, c_c])
                contact_points = np.append(contact_points, [[ri, ci, 2, offset, n]], axis=0)
                break
    # general case where -90 < theta < 90 ----------------------------------------------------
    else:
        tan_theta = np.tan(np.deg2rad(theta))
        dc = (1 + tan_theta ** 2) ** 0.5
        # find grasp line key point
        r_c = r0 - n / dc
        c_c = c0 + n * tan_theta / dc
        c_limit = int(round(w * np.cos(np.deg2rad(theta))))
        stop_point = -c_limit
        # ptt = time.time()
        # phase 1 find the left side contact ///////////////////////////////////////////////////
        for ldc in range(-c_limit, c_limit + 1):
            ci = int(round(c_c)) + ldc
            ri = gpl(r0, c0, theta, n, ci)
            if is_boundary_point([ri, ci], mask):
                offset = l2_distance([ri, ci], [r_c, c_c])
                contact_points = np.append(contact_points, [[ri, ci, 1, offset, n]], axis=0)
                stop_point = ldc
                break
        # print(f'phase 1 used {time.time()-ptt}')
        # phase 2 find the right side contact ///////////////////////////////////////////////////
        # ptt = time.time()
        for rdc in range(c_limit, stop_point, -1):
            ci = int(round(c_c)) + rdc
            ri = gpl(r0, c0, theta, n, ci)
            if is_boundary_point([ri, ci], mask):
                offset = l2_distance([ri, ci], [r_c, c_c])
                contact_points = np.append(contact_points, [[ri, ci, 2, offset, n]], axis=0)
                break
        # print(f'phase 2 used {time.time() - ptt}')
    if contact_points.shape[0] != 2:
        contact_points = -1 * np.ones(10).reshape((2, 5))
    return contact_points


def get_contact_points(r0, c0, sin_theta, cos_theta, hn, w, mask):
    dcr = hn * cos_theta
    dcc = hn * sin_theta
    r_c1 = r0 - dcr
    c_c1 = c0 + dcc
    r_c2 = r0 + dcr
    c_c2 = c0 - dcc

    start_d = [0, -6, 6, -10, 10, -3, 3, -8, 8, -4, 4, -12, 12, -2, 2, -13, 13, -11, 11, -1, 1, -9, 9, -5, 5, -7, 7]
    s1_found = -1
    s2_found = -1
    for s in start_d:
        both_starting_point_found = True
        dsr = s * sin_theta
        dsc = s * cos_theta

        if s1_found == -1:
            rt1 = r_c1 + dsr
            ct1 = c_c1 + dsc
            point_type1 = get_point_type([rt1, ct1], mask)
            if point_type1 == 1:
                s1_found = s
            else:
                both_starting_point_found = False

        if s2_found == -1:
            rt2 = r_c2 + dsr
            ct2 = c_c2 + dsc
            point_type2 = get_point_type([rt2, ct2], mask)
            if point_type2 == 1:
                s2_found = s
            else:
                both_starting_point_found = False
        if both_starting_point_found:
            break
    print('starting points', s1_found, s2_found)

    if s1_found != -1:
        l11 = -130
        l12 = s1_found
        r11 = s1_found
        r12 = 130
        left_contact_found = False
        right_contact_found = False
        while True:
            if not left_contact_found:
                l1m = (l11 + l12) / 2
                # calculate left test point
                l1rt = r_c1 + l1m * sin_theta
                l1ct = c_c1 + l1m * cos_theta
            if not right_contact_found:
                r1m = (r11 + r12) / 2
                # calculate right test point
                r1rt = r_c1 + r1m * sin_theta
                r1ct = c_c1 + r1m * cos_theta


# ---------------------------------------------------------------------------
def grasp_line_offset(r0, c0, theta, rp, cp):
    if abs(theta) == 90:
        offset = theta / 90 * (cp - c0)
    else:
        tan_theta = np.tan(np.deg2rad(theta))
        deno = (1 + tan_theta ** 2) ** 0.5
        offset = (r0 - tan_theta * c0 + tan_theta * cp - rp) / deno
    return offset


def get_contact_region(r0, c0, theta, outline_pixels, index_array):
    grasp_path_corners = get_grasp_rect_corners([r0, c0], theta, 19, 130)
    r_1 = int(round(np.amax(grasp_path_corners[0, :]))) - outline_pixels[0, 0]
    r_2 = int(round(np.amin(grasp_path_corners[0, :]))) - outline_pixels[0, 0]
    r_max = min(r_1 + 1, index_array.shape[0])
    r_min = max(0, r_2)
    print(r_1, r_2, index_array.shape, r_max, r_min)
    contact_region = np.ndarray((0, 2), dtype=np.uint8)
    for r in range(r_min, r_max):
        pixel_indices = index_array[r]
        p_a = outline_pixels[pixel_indices[0]: pixel_indices[1] + 1]
        for pixel in p_a:
            gpl_offset = grasp_line_offset(r0, c0, theta, pixel[0], pixel[1])
            if abs(gpl_offset) <= 19:
                # k_c1 = c0 + gpl_offset * np.sin(np.deg2rad(theta))
                # k_c2 = 130*np.cos(np.deg2rad(theta))
                # c_lep = k_c1 - k_c2
                # c_rep = k_c1 + k_c2
                # if c_lep <= pixel[1] <= c_rep:
                d_p0_squared = l2_distance(pixel, [r0, c0], 2)
                center_offset = (d_p0_squared - gpl_offset ** 2) ** 0.5
                if center_offset <= 130:
                    contact_region = np.append(contact_region, [[pixel[0], pixel[1]]], axis=0)
    return contact_region
