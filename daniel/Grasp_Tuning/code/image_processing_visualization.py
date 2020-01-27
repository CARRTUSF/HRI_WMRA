import numpy as np
import cv2


def feature_visualization(contact_region, normals, center, theta):  # center(row,col)
    """

    :param contact_region: nx[row,col,side]
    :param normals: nx[row0,col0,row1,col1]
    :param center: [row,col]
    :param theta: roll
    :return:
    """
    normal_sum = np.array([0.0, 0.0])
    vector_sum = np.array([0.0, 0.0])
    left_sum = np.array([0.0, 0.0])
    right_sum = np.array([0.0, 0.0])
    center = np.array(center)
    if contact_region.shape[0] > 50:
        for contact in contact_region:
            vector_sum += contact[0:2] - center
            if contact[2] == 0:
                # left side
                normal_sum += normals[contact[0], 0:2]
                left_sum += normals[contact[0], 0:2]
            else:
                # right side
                normal_sum += normals[contact[0], 2:4]
                right_sum += normals[contact[0], 2:4]
        r_normal = np.array([np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]) * 100
        l_normal = np.array([-np.sin(np.deg2rad(theta)), -np.cos(np.deg2rad(theta))]) * 100
        return normal_sum, vector_sum, left_sum, right_sum, l_normal, r_normal
    else:
        print(contact_region.shape)
        return None, None, None, None, None, None


def visualize_all(contact_region, normals, center, theta, img):
    normal_sum, vector_sum, left_sum, right_sum, l_normal, r_normal = feature_visualization(contact_region, normals,
                                                                                            center, theta)
    left_contact = np.ndarray((0, 2), dtype=np.uint8)
    right_contact = np.ndarray((0, 2), dtype=np.uint8)
    for i in range(contact_region.shape[0]):
        img[contact_region[i][0], contact_region[i][1]] = [255, 255, 0]
        if contact_region[i][2] == 0:
            left_contact = np.append(left_contact, contact_region[i][:2].reshape(1, 2), axis=0)
        else:
            right_contact = np.append(right_contact, contact_region[i][:2].reshape(1, 2), axis=0)

    print("-------------")
    print("normal sum", normal_sum)
    print("y loss", (normal_sum[0] ** 2 + normal_sum[1] ** 2)**0.5)
    print("vector sum", vector_sum)
    print("x loss", (vector_sum[0] ** 2 + vector_sum[1] ** 2)**0.5)
    print(left_sum)
    print(right_sum)
    print(l_normal)
    print(r_normal)
    alpha = np.rad2deg(np.arccos(np.dot(left_sum, l_normal) / (100 * (left_sum[0] ** 2 + left_sum[1] ** 2) ** 0.5)))
    beta = np.rad2deg(np.arccos(np.dot(right_sum, r_normal) / (100 * (right_sum[0] ** 2 + right_sum[1] ** 2) ** 0.5)))
    print('alpha', alpha)
    print('beta', beta)
    print('theta_loss', (alpha+beta)**2)
    print('theta_loss2', alpha**2 + beta**2)
    print("-------------")
    ml = int(left_contact.shape[0] / 2)
    mr = int(right_contact.shape[0] / 2)
    l_start = left_contact[ml]
    l_end = (l_start + left_sum).astype(int)
    r_start = right_contact[mr]
    r_end = (r_start + right_sum).astype(int)
    ns_end = (center + normal_sum).astype(int)
    vs_end = (center + vector_sum).astype(int)
    ln_end = (center + l_normal).astype(int)
    rn_end = (center + r_normal).astype(int)
    center = np.array(center)
    # normal vector of the contact region blue
    cv2.arrowedLine(img, (l_start[1], l_start[0]), (l_end[1], l_end[0]), (255, 0, 0))
    cv2.arrowedLine(img, (r_start[1], r_start[0]), (r_end[1], r_end[0]), (255, 0, 0))
    # normal sum green
    cv2.arrowedLine(img, (center[1], center[0]), (ns_end[1], ns_end[0]), (0, 255, 0))
    # vector sum red
    cv2.arrowedLine(img, (center[1], center[0]), (vs_end[1], vs_end[0]), (0, 0, 255))
    # gripper normals green+red
    cv2.arrowedLine(img, (center[1], center[0]), (ln_end[1], ln_end[0]), (0, 255, 255))
    cv2.arrowedLine(img, (center[1], center[0]), (rn_end[1], rn_end[0]), (0, 255, 255))
    cv2.imwrite('visualizationofall.png', img)
