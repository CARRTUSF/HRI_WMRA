import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from visulize_trajectory import save_path_image, save_loss_image
from image_processing_visualization import visualize_all
import time


def get_outline(obj_mask):
    """
    
    :param obj_mask: image mask
    :return:
        outline: binary image of the image outline
        outline_pixel: pixel indices of the outline
            nx[col1, col2] n is the row index col1 is the left outline point col2 is the right one
        center_line: nx[row,col_c]
        center: center of the outline [row, col]

    """
    r, c = obj_mask.shape
    outline_pixel = -1 * np.ones((r, 2), np.uint8)
    # center_line = np.ndarray((0, 2), dtype=np.uint8)
    outline = np.zeros((r, c), np.uint8)
    # get outline
    for i in range(r):
        for j in range(c):
            if obj_mask[i, j] == 1:
                outline_pixel[i, 0] = j
                break
        if outline_pixel[i, 0] == -1:
            pass
        else:
            for k in range(c):
                if obj_mask[i, c - k - 1] == 1:
                    outline_pixel[i, 1] = c - k - 1
                    break
        # # get center line
        # if outline_pixel[i, 1] != -1:   # if outline_pixel[i,1]!=-1 then outline_pixel[i,0] must not be -1
        #     print([[i, int((outline_pixel[i, 0] + outline_pixel[i, 1])/2)]])
        #     center_line = np.append(center_line, [[i, int((outline_pixel[i, 0] + outline_pixel[i, 1])/2)]], axis=0)
    # get outline image
    row_sum = 0
    row_count = 0
    col_sum = 0
    for n in range(outline_pixel.shape[0]):
        if outline_pixel[n, 0] != -1:
            outline[n, outline_pixel[n, 0]] = 1
            outline[n, outline_pixel[n, 1]] = 1
            row_sum += n
            row_count += 1
            col_sum += (outline_pixel[n, 0] + outline_pixel[n, 1])
    center = np.array([row_sum / row_count, col_sum / (2 * row_count)])
    return outline, outline_pixel, center


def find_edge_neighbors(outline_image, edge_point, window_size):
    """

    :param outline_image: binary outline image of the object
    :param edge_point: the point of interest (PoI) [row, col]
    :param window_size: neighborhood size
    :return:
        edge_neighbors: outline points in the neighborhood of PoI nx[row, col]
    """
    edge_neighbors = []
    r = int((window_size - 1) / 2)
    for i in range(window_size):
        for j in range(window_size):
            if outline_image[edge_point[0] - r + i, edge_point[1] - r + j] == 1:
                if i == r and j == r:
                    pass
                else:
                    edge_neighbors.append([edge_point[0] - r + i, edge_point[1] - r + j])
    edge_neighbors = np.array(edge_neighbors)
    return edge_neighbors


def find_pixel_normals(outline, outline_pixels, center, window_size):
    """

    :param outline: binary outline image of the object
    :param outline_pixels: pixel indices of the outline
            nx[col1, col2] n is the row index col1 is the left outline point col2 is the right one
    :param center: center of the outline [row, col]
    :param window_size: neighborhood size for find edge normal
    :return:
        pixel_normals: nx4 array n is the row, :2 normal of the outline_pixel[n][0], 2:4 normal of the outline_pixel[n][1]
        nx[row0,col0,row1,col1]
    """
    r, c = outline.shape
    pixel_normals = -2 * np.ones((r, 4))
    pad_n = int((window_size - 1) / 2)
    padded_outline = pad_zeros(outline, pad_n)
    padded_center = np.array([center[0] + pad_n, center[1] + pad_n])    # [row col]
    for i in range(r):
        if outline_pixels[i, 0] != -1:
            # print('start')
            for j in range(2):  # j = 0 left side outline, 1 right side outline
                outline_pixel = np.array([i + pad_n, outline_pixels[i, j] + pad_n])     # [row col]
                # print(outline_pixel)
                neighbors = find_edge_neighbors(padded_outline, outline_pixel, window_size)     # [row col]
                # print(neighbors)
                if neighbors.shape[0] == 0:
                    continue
                else:
                    normals_sum = np.array([0.0, 0.0])
                    for neighbor in neighbors:      # [row col]
                        n_of_e_neighbor = find_normal_vectors(neighbor, outline_pixel,
                                                              outline_pixel[0] - padded_center[0], j)
                        # print('neighbor normal')
                        # print(outline_pixel[0]-padded_center[0], j)
                        # print(n_of_e_neighbor)
                        normals_sum += n_of_e_neighbor      # [row col]
                    # print('sum of neighbors normal')
                    # print(normals_sum)
                    # normals = np.array(normals)
                    # outward_normal = select_outward_normal(normals, [padded_center[1], padded_center[0]], [outline_pixel[1], outline_pixel[0]])
                    # print(outward_normal)
                    normal = normalize(normals_sum)
                    pixel_normals[i, 2 * j] = normal[0]
                    pixel_normals[i, 2 * j + 1] = normal[1]
    return pixel_normals


def normalize(n):
    norm = (n[0] ** 2 + n[1] ** 2) ** 0.5
    n[0] /= norm
    n[1] /= norm
    return n


def length_between_points(p1, p2):
    length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return length


def find_normal_vectors(a, b, height, side):
    """

    :param a: point a [row, col]
    :param b: point b [row, col]
    :param height: height of the point b relative to the outline center
    :param side: point b is at left(0) or right(1) side of the center
    :return:
        normal: the normal vector of neighbor vector b->a. [row, col]([normal_y, normal_x])
    """
    y = a[0] - b[0]
    x = a[1] - b[1]
    if y == 0:
        if height > 0:
            normal = [0, 1]
        else:
            normal = [0, -1]
    else:
        nx = abs(y / (x ** 2 + y ** 2) ** 0.5)
        ny = - x / y * nx
        if side == 1:
            normal = [ny, nx]
        else:
            normal = [-ny, -nx]
    normal = np.array(normal)
    return normal


def pad_zeros(img, x):
    r, c = img.shape
    padded_img = np.zeros((r + 2 * x, c + 2 * x), np.uint8)
    padded_img[x:x + r, x:x + c] = img
    return padded_img


def normal_end(outline_pixels, normals):
    """

    :param outline_pixels: nx[col0, col1]
    :param normals: nx[row0,col0,row1,col1]
    :return:
        normal_vectors nx[r_end, c_end, r_start, c_start]
    """
    normal_vectors = []
    for i in range(len(normals)):
        if normals[i, 0] != -2:
            end1 = [normals[i, 0] * 5 + i, normals[i, 1] * 5 + outline_pixels[i, 0], i, outline_pixels[i, 0]]
            normal_vectors.append(end1)
        if normals[i, 2] != -2:
            end2 = [normals[i, 2] * 5 + i, normals[i, 3] * 5 + outline_pixels[i, 1], i, outline_pixels[i, 1]]
            normal_vectors.append(end2)
    return normal_vectors


def sqr_norm_of_normals_sum(normals, center_y):
    """
    old
    @@@@@@@@@@@@@@@@@@@@@@@@@
    :param normals:
    :param center_y:
    :return:
    """
    s = int(round(center_y - 18))
    e = int(round(center_y + 18))
    normal_sum = [0, 0]
    has_contact = False
    for i in range(s, e + 1):
        if normals[i][0] != -2:
            has_contact = True
            normal_sum[0] += normals[i][0]
            normal_sum[1] += normals[i][1]
        if normals[i][2] != -2:
            has_contact = True
            normal_sum[0] += normals[i][2]
            normal_sum[1] += normals[i][3]
    if has_contact:
        sqr_norm = normal_sum[0] ** 2 + normal_sum[1] ** 2
    else:
        sqr_norm = - 100
    return sqr_norm


def sqr_norm_contact_vectors(outline_pixels, center):
    """
    old
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    :param outline_pixels:
    :param center:
    :return:
    """
    s = int(round(center[1] - 18))
    e = int(round(center[1] + 18))
    vector_sum = [0, 0]
    has_contact = False
    for i in range(s, e + 1):
        for j in range(2):
            if outline_pixels[i][j] != -1:
                has_contact = True
                vector_sum[0] += (outline_pixels[i][j] - center[0]) / 1000
                vector_sum[1] += (i - center[1]) / 1000
    if has_contact:
        sqr_norm = vector_sum[0] ** 2 + vector_sum[1] ** 2
    else:
        sqr_norm = -100
    return sqr_norm


def contact_angle_sum(outline_pixels, normals, center, theta, option):
    """
    old
    @@@@@@@@@@@@@@@@@@@@@@@@@@
    :param outline_pixels:
    :param normals:
    :param center:
    :param theta:
    :param option:
    :return:
    """
    min_y = center[0] - 18 / np.cos(np.deg2rad(theta))
    max_y = center[0] + 18 / np.cos(np.deg2rad(theta))
    r, c = outline_pixels.shape
    contact_region = -1 * np.ones((r, c), np.uint8)
    left = np.array([0.0, 0.0])
    right = np.array([0.0, 0.0])
    for i in range(outline_pixels.shape[0]):
        for j in range(2):
            if outline_pixels[i, j] != -1:
                # print(i, outline_pixels[i, j])
                # print(theta)
                # print(np.tan(np.deg2rad(theta)))
                # print(center)
                y = i - np.tan(np.deg2rad(theta)) * (outline_pixels[i, j] - center[1])
                if min_y <= y <= max_y:
                    contact_region[i, j] = outline_pixels[i, j]
                    if j == 0:
                        left += normals[i, 0:2]
                    else:
                        right += normals[i, 2:4]
    # print(left)
    # print(right)
    alpha = np.rad2deg(np.arctan2(left[1], left[0]))
    if alpha < 0:
        # print("----------------------")
        # print(alpha)
        alpha += 180
    else:
        alpha -= 180
    beta = np.rad2deg(np.arctan2(right[1], right[0]))
    # print(theta, alpha, beta)
    angle_sum = np.abs(alpha - theta) + np.abs(beta - theta)
    if option == 1:
        return contact_region
    else:
        return angle_sum


def get_contact_region(outline_pixels, gripper_center, theta):  # gripper_center(row, col)
    """

    :param outline_pixels: nx[col0,col1]
    :param gripper_center: [row,col] gripper center
    :param theta: gripper roll
    :return:
        contact_region: nx[row, col, side]
    """
    min_y = gripper_center[0] - 18 / np.cos(np.deg2rad(theta))
    max_y = gripper_center[0] + 18 / np.cos(np.deg2rad(theta))
    contact_region = []
    s_time = time.time()
    for i in range(outline_pixels.shape[0]):
        # start_time = time.time()
        for j in range(2):
            if outline_pixels[i, j] != -1:
                y = i - np.tan(np.deg2rad(theta)) * (outline_pixels[i, j] - gripper_center[1])
                if min_y <= y <= max_y:
                    contact_region.append([i, outline_pixels[i, j], j])
        # print("test two outline pixels used", time.time()-start_time, "seconds")
    print("get contact region used", time.time() - s_time, "seconds", len(contact_region))
    contact_region = np.array(contact_region)   # (row, col, j)
    return contact_region


def total_loss(contact_region, normals, gripper_center, theta):    # gripper_center(row,col)
    """

    :param contact_region: nx[row,col,side]
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
    if contact_region.shape[0] > 50:
        for contact in contact_region:
            vector_sum += contact[0:2] - gripper_center       # [row, col]
            if contact[2] == 0:
                # left side
                normal_sum += normals[contact[0], 0:2]      # [row, col]
                left_sum += normals[contact[0], 0:2]        # [row, col]
            else:
                # right side
                normal_sum += normals[contact[0], 2:4]      # [row, col]
                right_sum += normals[contact[0], 2:4]       # [row, col]
        x_loss = (vector_sum[0] ** 2 + vector_sum[1] ** 2)**0.5
        y_loss = (normal_sum[0] ** 2 + normal_sum[1] ** 2)**0.5
        r_normal = np.array([np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))])    # [row, col]
        l_normal = np.array([-np.sin(np.deg2rad(theta)), -np.cos(np.deg2rad(theta))])   # [row, col]
        if left_sum.all() == 0:
            alpha = 30
            beta = 30
        elif right_sum.all() == 0:
            beta = 30
            alpha = 30
        else:
            c_alpha = np.dot(l_normal, left_sum) / (left_sum[0]**2+left_sum[1]**2)**0.5
            c_beta = np.dot(r_normal, right_sum) / (right_sum[0]**2 + right_sum[1]**2)**0.5
            alpha = np.rad2deg(np.arccos(c_alpha))
            beta = np.rad2deg(np.arccos(c_beta))
        theta_loss = (alpha + beta)**2
        return x_loss, y_loss, theta_loss
    else:
        return -1, -1, -1


def get_loss_surface(outline_pixels, normals, theta):
    """

    :param outline_pixels: nx[col0,col1]
    :param normals: nx[row0,col0,row1,col1]
    :param theta: gripper roll
    :return:
        normalized_loss_surface
    """
    local_loss_surface = []
    for i in range(0, 60):
        x = 375 * i / 59 + 450
        for j in range(0, 60):
            y = 175 * j / 59 + 250
            next_contact = get_contact_region(outline_pixels, [y, x], theta)
            x_loss, y_loss, theta_loss = total_loss(next_contact, normals, [y, x], theta)
            local_loss_surface.append([x, y, x_loss, y_loss, theta_loss])
    local_loss_surface = np.array(local_loss_surface)
    for c in range(3):
        max_loss = np.amax(local_loss_surface[:, c + 2])
        local_loss_surface[:, c + 2] /= max_loss
    loss = (local_loss_surface[:, 2] + local_loss_surface[:, 3] + local_loss_surface[:, 4])/3
    for k in range(local_loss_surface.shape[0]):
        if local_loss_surface[k, 2] < 0:
            loss[k] = 1
    normalized_loss_surface = np.stack((local_loss_surface[:, 0], local_loss_surface[:, 1], loss), 1)
    return normalized_loss_surface


def get_local_loss_surface(outline_pixels, normals, current_pos, next_theta):
    local_loss_surface = []
    for i in range(0, 60):
        dx = 120 * i / 59 - 40
        for j in range(0, 60):
            dy = 120 * j / 59 - 60
            # ri = r * np.sqrt(np.random.random())
            # theta = np.random.random() * 2 * np.pi
            # next_x = current_pos[0] + ri * np.cos(theta)
            # next_y = current_pos[1] + ri * np.sin(theta)
            next_x = current_pos[0] + dx
            next_y = current_pos[1] + dy
            next_contact = get_contact_region(outline_pixels, [next_y, next_x], next_theta)
            x_loss, y_loss, theta_loss = total_loss(next_contact, normals, [next_y, next_x], next_theta)
            local_loss_surface.append([next_y, next_x, x_loss, y_loss, theta_loss])
    local_loss_surface = np.array(local_loss_surface)
    for c in range(3):
        max_loss = np.amax(local_loss_surface[:, c + 2])
        local_loss_surface[:, c + 2] /= max_loss
    loss = (local_loss_surface[:, 2] + local_loss_surface[:, 3] + local_loss_surface[:, 4])/3
    for k in range(local_loss_surface.shape[0]):
        if local_loss_surface[k, 2] < 0:
            loss[k] = 1
    normalized_loss_surface = np.stack((local_loss_surface[:, 0], local_loss_surface[:, 1], loss), 1)
    return normalized_loss_surface


def get_local_theta_loss(outline_pixels, normals, current_pos):
    local_loss_theta = []
    for i in range(160):
        next_theta = i - 80
        next_x = current_pos[0]
        next_y = current_pos[1]
        next_contact = get_contact_region(outline_pixels, [next_y, next_x], next_theta)
        x_loss, y_loss, theta_loss = total_loss(next_contact, normals, [next_y, next_x], next_theta)
        local_loss_theta.append([next_y, next_x, next_theta, x_loss, y_loss, theta_loss])
    local_loss_theta = np.array(local_loss_theta)
    return local_loss_theta


def find_next_gripper_pose(outline_pixels, normals, current_pos):
    r = 8
    threshold = 0.1
    min_loss = current_pos[3]
    next_pos = current_pos
    for i in range(50):
        ri = r * np.sqrt(np.random.random())
        phi = np.random.random() * 2 * np.pi
        next_x = current_pos[0] + ri * np.cos(phi)
        next_y = current_pos[1] + ri * np.sin(phi)
        for k in range(30):
            # s_time = time.time()
            if k < 2:
                next_theta = current_pos[2] * k
            else:
                d_theta = 90 * np.random.random() - 45
                next_theta = current_pos[2] + d_theta
            # start_time = time.time()
            next_contact = get_contact_region(outline_pixels, [next_y, next_x], next_theta)
            # print("get contact region used", time.time()-start_time, "seconds")
            # start_time = time.time()
            x_loss, y_loss, theta_loss = total_loss(next_contact, normals, [next_y, next_x], next_theta)
            loss = x_loss + y_loss + theta_loss
            # print("find current loss used", time.time()-start_time, "seconds")
            if loss != -3 and min_loss - loss > threshold:
                min_loss = loss
                next_pos = [next_x, next_y, next_theta, loss, x_loss, y_loss, theta_loss]
            # print("one search used", time.time()-s_time, "seconds")
    return next_pos


def find_gripper_trajectory(outline_pixels, normals, current_pos):
    iterations = 0
    if current_pos[3] == -1:
        current_contact = get_contact_region(outline_pixels, [current_pos[1], current_pos[0]], current_pos[2])
        l1, l2, l3 = total_loss(current_contact, normals, [current_pos[1], current_pos[0]], current_pos[2])
        current_loss = l1 + l2 + l3
        current_pos[3:7] = [current_loss, l1, l2, l3]
    trajectory = [current_pos]
    while iterations < 20:
        start_time = time.time()
        iterations += 1
        next_pos = find_next_gripper_pose(outline_pixels, normals, current_pos)
        if current_pos == next_pos:
            break
        else:
            current_pos = next_pos
            trajectory.append(current_pos)
        print("iteration", iterations, "used", time.time()-start_time, "seconds")
    trajectory = np.array(trajectory)
    return trajectory


def test():
    path = os.path.dirname(os.getcwd())
    image_name = 'spoon3_mask.png'
    image2_name = 'spoon3.png'
    img = cv2.imread(os.path.join(path, 'pictures', image_name))
    print(img.shape)
    outline, outline_pixels, center = get_outline(img[:, :, 0])  # center: outline center [row, col]
    # print(outline)
    # print(outline_pixels)
    # print(center)
    # ip = world2image(0.012, -0.03825, 0.142345)
    # print("image point")
    # print(ip)
    normals = find_pixel_normals(outline, outline_pixels, center, 7)        # nx[row0,col0,row1,col1]
    # print(normals)

    # loss_surface_points = get_loss_surface(outline_pixels, normals, 0)
    # loss_surface_points = get_local_loss_surface(outline_pixels, normals, [659, 314], 0)
    # loss_vs_param= get_local_theta_loss(outline_pixels, normals, [659, 314, 0, -1])

    # -----------------------------------------------------------------------------------------------------------
    trajectory = find_gripper_trajectory(outline_pixels, normals, [659, 314, 0, -1, -1, -1, -1])
    print(trajectory)

    sqr_norm_vs_y = []
    sqr_norm_vs_x = []
    ang_diff_vs_theta = []
    for i in range(200, 600):
        sqr_norm = sqr_norm_of_normals_sum(normals, i)
        sqr_norm_vs_y.append([i, sqr_norm])
        sqr_norm_x = sqr_norm_contact_vectors(outline_pixels, [259 + i, 314])
        sqr_norm_vs_x.append([259 + i, sqr_norm_x])
    for ang in range(-30, 31):
        ang_diff = contact_angle_sum(outline_pixels, normals, center, ang, 0)
        ang_diff_vs_theta.append([ang, ang_diff])

    normal_vectors = normal_end(outline_pixels, normals)        # nx[r_end, c_end, r_start, c_start]
    contact_region = get_contact_region(outline_pixels, center, -30)
    contact_region_img = np.zeros((img.shape[0:2]))
    for contact in contact_region:
        contact_region_img[contact[0], contact[1]] = 1
    # for r in range(contact_region_img.shape[0]):
    #     for c in range(contact_region_img.shape[1]):
    #         for m in range(2):
    #             if contact_region[r, m] != -1:
    #                 contact_region_img[r, contact_region[r, m]] = 1

    plt.figure(1)
    plt.title('Outline of the object')
    plt.imshow(outline)
    plt.scatter(center[1], center[0], color='r')
    for n_vector in normal_vectors:
        plt.plot([n_vector[1], n_vector[3]], [n_vector[0], n_vector[2]], color='g')
    sqr_norm_vs_y = np.array(sqr_norm_vs_y)
    sqr_norm_vs_x = np.array(sqr_norm_vs_x)
    ang_diff_vs_theta = np.array(ang_diff_vs_theta)
    plt.figure(2)
    plt.xlabel('Gripper y position(up-down in world coordinates)')
    plt.ylabel('Squared norm of the sum of the normal of each contact region pixel')
    plt.title('Grasp feature for gripper up-down adjustment')
    plt.plot(sqr_norm_vs_y[:, 0], sqr_norm_vs_y[:, 1])
    # plt.plot(loss_surface_points[:, 1], loss_surface_points[:, 3])
    plt.figure(3)
    plt.xlabel('Gripper x position(left-right in world coordinates)')
    plt.ylabel('Squared norm of the sum of the contact vectors')
    plt.title('Grasp feature for gripper left-right adjustment')
    plt.plot(sqr_norm_vs_x[:, 0], sqr_norm_vs_x[:, 1])
    # plt.plot(loss_surface_points[:, 0], loss_surface_points[:, 2])
    plt.figure(4)
    plt.xlabel('Gripper angle about its z-axis')
    plt.ylabel('Contact angle(degrees)')
    plt.title('Grasp feature for gripper roll adjustment')
    plt.plot(ang_diff_vs_theta[:, 0], ang_diff_vs_theta[:, 1])
    plt.figure(5)
    plt.title('Mask of the object from mask R-CNN')
    plt.imshow(img[:, :, 0])

    plt.figure(6)
    plt.title('contact region')
    plt.imshow(contact_region_img)

    # fig = plt.figure(7)
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax1.scatter(loss_surface_points[:, 0], loss_surface_points[:, 1], loss_surface_points[:, 2], marker='x')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.set_zlabel('Total loss(theta = 0 degrees')

    img2 = cv2.imread(os.path.join(path, 'pictures', image2_name))
    save_path_image(trajectory, img2, r'C:\Users\75077\OneDrive\2019\finalproject\spoon1')
    # save_loss_image(trajectory[:, 3])

    save_path_image(trajectory, img*255, r'C:\Users\75077\OneDrive\2019\finalproject\spoon1\mask')
    center_v = [314, 659]  # [row col]
    theta_v = 30
    contact_region_v = get_contact_region(outline_pixels, center_v, theta_v)
    # print(contact_region_v)
    visualize_all(contact_region_v, normals, center_v, theta_v, img)

    plt.show()


if __name__ == '__main__':
    test()
