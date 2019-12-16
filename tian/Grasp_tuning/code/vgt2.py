import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from visulize_trajectory import save_path_image, save_loss_image, get_corners
from image_processing_visualization import visualize_all
import time
from vision4grasp_tuning import find_pixel_normals, get_outline


next_pose1 = []


def neighbours(x, y, img):
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1, y], img[x_1, y1], img[x, y1], img[x1, y1],  # P2,P3,P4,P5
            img[x1, y], img[x1, y_1], img[x, y_1], img[x_1, y_1]]  # P6,P7,P8,P9


def transitions(sk_neighbours):
    n = sk_neighbours + sk_neighbours[0:1]  # P2, P3, ... , P8, P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


def zhangsuen(mask_pixels, image):
    image_thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1  # the points to be removed (set as 0)
    skeleton_pixels = mask_pixels.copy()
    while changing1 or changing2:  # iterates until no further changes occur in the image
        new_pixels = []
        # Step 1
        changing1 = []
        # rows, columns = image_thinned.shape  # x for rows, y for columns
        # for x in range(1, rows - 1):  # No. of  rows
        #     for y in range(1, columns - 1):  # No. of columns
        for pixel in skeleton_pixels:
            x = pixel[0]
            y = pixel[1]
            P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image_thinned)
            if (image_thinned[x, y] == 1 and  # Condition 0: Point P1 in the object regions
                    2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and  # Condition 2: S(P1)=1
                    P2 * P4 * P6 == 0 and  # Condition 3
                    P4 * P6 * P8 == 0):  # Condition 4
                changing1.append((x, y))
            else:
                new_pixels.append([x, y])
        for x, y in changing1:
            image_thinned[x, y] = 0
        skeleton_pixels = new_pixels.copy()
        new_pixels = []
        # Step 2
        changing2 = []
        # for x in range(1, rows - 1):
        #     for y in range(1, columns - 1):
        for pixel in skeleton_pixels:
            x = pixel[0]
            y = pixel[1]
            P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image_thinned)
            if (image_thinned[x, y] == 1 and  # Condition 0
                    2 <= sum(n) <= 6 and  # Condition 1
                    transitions(n) == 1 and  # Condition 2
                    P2 * P4 * P8 == 0 and  # Condition 3
                    P2 * P6 * P8 == 0):  # Condition 4
                changing2.append((x, y))
            else:
                new_pixels.append([x, y])
        for x, y in changing2:
            image_thinned[x, y] = 0
        skeleton_pixels = new_pixels.copy()
    return skeleton_pixels      # image_thinned


def current_pos2skeleton_points_distance(current_pos, sk_points):
    distances = []
    for i in range(len(sk_points)):
        distance = ((sk_points[i][0]-current_pos[0])**2 + (sk_points[i][1]-current_pos[1])**2)**0.5
        distances.append([i, distance])
    distances = sorted(distances, key=lambda l:l[1])
    return distances


def get_outline_normal(mask, outline_image, r_crop, c_crop, window_size):
    x = int(window_size/2)
    outline_pixels = np.ndarray((0, 2), dtype=np.uint8)
    outline_normals = np.ndarray((0, 2), dtype=np.uint8)
    padded_mask = pad_zeros(mask, x)
    r, c = outline_image.shape
    for i in range(r):
        for j in range(c):
            if outline_image[i, j] == 1:
                outline_pixels = np.append(outline_pixels, [[i+r_crop, j+c_crop]], axis=0)
                v = np.arange(-x, x+1)
                neighbor_vectors_row = np.repeat(v.reshape((-1, 1)), window_size, axis=1)
                neighbor_vectors_col = np.repeat(v.reshape((1, -1)), window_size, axis=0)
                window_mask = padded_mask[i:i+2*x+1, j:j+2*x+1]
                occupied_neighbor_vectors_row = neighbor_vectors_row * window_mask
                occupied_neighbor_vectors_col = neighbor_vectors_col * window_mask
                n_r = np.sum(occupied_neighbor_vectors_row)
                n_c = np.sum(occupied_neighbor_vectors_col)
                neighbor_vectors_sum = normalize([n_r, n_c])
                normal = - np.array(neighbor_vectors_sum).reshape(1, 2)
                outline_normals = np.append(outline_normals, normal, axis=0)
    return outline_pixels, outline_normals


def get_outline_and_normal(mask, r_crop, c_crop, window_size):
    x = int(window_size/2)
    outline_pixels = np.ndarray((0, 2), dtype=np.uint8)
    outline_normals = np.ndarray((0, 2), dtype=np.uint8)
    padded_mask = pad_zeros(mask, x)
    r, c = mask.shape
    # ----------------construct index array
    current_index = -1
    current_row = -1
    ix1 = 0
    ix2 = 0
    index_array = np.ndarray((0, 2), dtype=np.uint8)
    # -----------------
    for i in range(r):
        for j in range(c):
            if mask[i, j] == 1:
                boundary_detector = padded_mask[i+x-1, j+x]*padded_mask[i+x+1, j+x]*padded_mask[i+x, j+x-1]*padded_mask[i+x, j+x+1]
                if boundary_detector == 0:
                    outline_pixels = np.append(outline_pixels, [[i+r_crop, j+c_crop]], axis=0)
                    # ---------------------------
                    current_index += 1
                    if current_row == -1:
                        current_row = i
                    if i == current_row:
                        ix2 = current_index
                    else:
                        index_array = np.append(index_array, [[ix1, ix2]], axis=0)
                        ix1 = current_index
                        ix2 = current_index
                        current_row = i
                    # ------------------------
                    v = np.arange(-x, x+1)
                    neighbor_vectors_row = np.repeat(v.reshape((-1, 1)), window_size, axis=1)
                    neighbor_vectors_col = np.repeat(v.reshape((1, -1)), window_size, axis=0)
                    window_mask = padded_mask[i:i+2*x+1, j:j+2*x+1]
                    occupied_neighbor_vectors_row = neighbor_vectors_row * window_mask
                    occupied_neighbor_vectors_col = neighbor_vectors_col * window_mask
                    n_r = np.sum(occupied_neighbor_vectors_row)
                    n_c = np.sum(occupied_neighbor_vectors_col)
                    neighbor_vectors_sum = normalize([n_r, n_c])
                    normal = - np.array(neighbor_vectors_sum).reshape(1, 2)
                    outline_normals = np.append(outline_normals, normal, axis=0)
    # --------------
    index_array = np.append(index_array, [[ix1, ix2]], axis=0)
    return outline_pixels, outline_normals, index_array


def normalize(n):
    norm = (n[0] ** 2 + n[1] ** 2) ** 0.5
    n[0] /= norm
    n[1] /= norm
    return n


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
    print(len(outline_pixels))
    print(len(normals))
    normal_vectors = []
    for i in range(len(normals)):
            end = [normals[i, 0] * 5 + outline_pixels[i, 0], normals[i, 1] * 5 + outline_pixels[i, 1], outline_pixels[i, 0], outline_pixels[i, 1]]
            normal_vectors.append(end)
    return normal_vectors


def find_contact_region(outline_pixels, index_array, center, theta):
    """

    :param outline_pixels:
    :param index_array:
    :param center: [row,col]
    :param theta:
    :return:
    """
    grasp_path_corners = get_corners(center, theta, 19, 130)
    r_1 = int(round(np.amax(grasp_path_corners[0, :]))) - outline_pixels[0, 0]
    r_2 = int(round(np.amin(grasp_path_corners[0, :]))) - outline_pixels[0, 0]
    r_max = min(r_1, index_array.shape[0])
    r_min = max(0, r_2)
    contact_region = np.ndarray((0, 2), dtype=np.uint8)

    # --------------------------------------find contact seeds
    contact_seeds_found = False
    contact_seeds = np.array([[-1, -1], [-1, -1]])
    seeding_sample_size = 10
    while not contact_seeds_found:
        search_sample = np.random.randint(r_min, r_max, seeding_sample_size)
        for k in range(seeding_sample_size):
            row_index = search_sample[k]
            pixel_indices = index_array[row_index]
            p_a = outline_pixels[pixel_indices[0]: pixel_indices[1]+1]
            for i in range(len(p_a)):
                if inside_grasp_path(center, theta, p_a[i]):
                    col_average = np.sum(p_a[:, 1])/len(p_a)
                    if p_a[i, 1] < col_average:
                        contact_seeds[0] = [row_index, i]
                        contact_region = np.append(contact_region, [[i+pixel_indices[0], 0]], axis=0)
                    else:
                        contact_seeds[1] = [row_index, i]
                        contact_region = np.append(contact_region, [[i+pixel_indices[0], 1]], axis=0)
            if contact_seeds[0, 0] != -1 and contact_seeds[1, 0] != -1:
                contact_seeds_found = True
                break
    # ---------------------------------------find contact region
    search_indicator = [0, 0, 0, 0]
    # index of index_array
    search_rows = np.array([contact_seeds[0, 0], contact_seeds[0, 0], contact_seeds[1, 0], contact_seeds[1, 0]])
    row_change = np.array([1, -1, 1, -1])
    while True:
        search_rows += row_change
        for i in range(4):
            if not search_indicator[i]:
                stopping_row = True
                if search_rows[i]>len(index_array)-1 or search_rows[i]<0:
                    pass
                    #print(search_rows[i])
                else:
                    p_i = index_array[search_rows[i]]
                    p_a = outline_pixels[p_i[0]: p_i[1]+1]
                    col_avg = np.sum(p_a[:, 1])/len(p_a)
                    for j in range(len(p_a)):
                        if i < 2:
                            if p_a[j, 1] - col_avg < 0 and inside_grasp_path(center, theta, p_a[j]):
                                stopping_row = False
                                contact_region = np.append(contact_region, [[p_i[0]+j, 0]], axis=0)
                        else:
                            if p_a[j, 1] - col_avg > 0 and inside_grasp_path(center, theta, p_a[j]):
                                stopping_row = False
                                contact_region = np.append(contact_region, [[p_i[0]+j, 1]], axis=0)
                if stopping_row:
                    search_indicator[i] = 1
        if search_indicator[0]*search_indicator[1]*search_indicator[2]*search_indicator[3] == 1:
            break

    return contact_region


def inside_grasp_path(center, theta, pixel):
    min_y = center[0] - 19 / np.cos(np.deg2rad(theta))
    max_y = center[0] + 19 / np.cos(np.deg2rad(theta))
    y = pixel[0] - np.tan(np.deg2rad(theta)) * (pixel[1] - center[1])
    if min_y <= y <= max_y:
        return True
    else:
        return False


def total_loss(contact_region, outline_pixels, normals, gripper_center, theta):    # gripper_center(row,col)
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
    if contact_region.shape[0] > 70:
        for contact_marker in contact_region:
            contact = outline_pixels[contact_marker[0]]
            vector_sum += contact - gripper_center       # [row, col]
            normal_sum += normals[contact_marker[0]]
            if contact_marker[1] == 0:
                # left side
                left_sum += normals[contact_marker[0]]        # [row, col]
            else:
                # right side
                right_sum += normals[contact_marker[0]]       # [row, col]
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


def find_next_gripper_pose(outline_pixels, index_array, normals, current_pos, angle_search_space):
    r = 8
    threshold = 0.1
    min_loss = current_pos[3]
    next_pos = current_pos
    for i in range(20):
        ri = r * np.sqrt(np.random.random())
        phi = np.random.random() * 2 * np.pi
        next_row = current_pos[0] + ri * np.cos(phi)
        next_col = current_pos[1] + ri * np.sin(phi)
        for k in range(len(angle_search_space)):
            if k < 8:
                next_theta = angle_search_space[k] + current_pos[2]
            else:
                next_theta = angle_search_space[k]
            # start_time = time.time()
            next_contact = find_contact_region(outline_pixels, index_array, [next_row, next_col], next_theta)
            # print("get contact region used", time.time()-start_time, "seconds")
            # start_time = time.time()
            x_loss, y_loss, theta_loss = total_loss(next_contact, outline_pixels, normals, [next_row, next_col], next_theta)
            loss = x_loss + y_loss + theta_loss
            # print("find current loss used", time.time()-start_time, "seconds")
            if loss != -3 and min_loss - loss > threshold:
                min_loss = loss
                next_pos = [next_row, next_col, next_theta, loss, x_loss, y_loss, theta_loss]
            # print("one search used", time.time()-s_time, "seconds")
    return next_pos


def find_gripper_trajectory(outline_pixels, index_array, normals, mask_skeleton, current_pos):
    iterations = 0
    distance_array = current_pos2skeleton_points_distance(current_pos[0:2], mask_skeleton)
    closest_skp = [mask_skeleton[distance_array[0][0]][0], mask_skeleton[distance_array[0][0]][1]]
    if current_pos[3] == -1:
        current_pos[0] = closest_skp[0]
        current_pos[1] = closest_skp[1]
        current_contact = find_contact_region(outline_pixels, index_array, [current_pos[0], current_pos[1]], current_pos[2])
        l1, l2, l3 = total_loss(current_contact, outline_pixels, normals, [current_pos[0], current_pos[1]], current_pos[2])
        current_loss = l1 + l2 + l3
        current_pos[3:7] = [current_loss, l1, l2, l3]
    trajectory = [current_pos]
    # angle_search_space = np.arange(-90, 96, 6)
    # angle_search_space = np.append([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], angle_search_space)
    angle_search_space = np.arange(-30, 30, 5)
    angle_search_space = np.append([-4, -3, -2, -1, 1, 2, 3, 4], angle_search_space)
    while iterations < 20:
        start_time = time.time()
        iterations += 1
        next_pos = find_next_gripper_pose(outline_pixels, index_array, normals, current_pos,angle_search_space)
        if current_pos == next_pos:
            break
        else:
            current_pos = next_pos
            trajectory.append(current_pos)
        print("iteration", iterations, "used", time.time() - start_time, "seconds")
    trajectory = np.array(trajectory)
    return trajectory


def test1():
    path = os.path.dirname(os.getcwd())
    mask_image_name = 'spoon3_mask.png'
    image_name = 'spoon3.png'
    mask_img = cv2.imread(os.path.join(path, 'pictures', mask_image_name))
    print(mask_img.shape)
    roi = mask_img[150:570, 540:740, 0]
    #start_time1 = time.time()
    outline, outline_pixels, center = get_outline(roi)
    normals = find_pixel_normals(outline, outline_pixels, center, 7)
    #time1 = time.time() - start_time1

    #start_time2 = time.time()
    edge = cv2.Canny(roi * 255, 100, 200)
    edge = (edge / 255).astype(np.uint8)
    outline_pixels_new, outline_normals = get_outline_normal(roi, edge, 150, 540, 7)
    #time2 = time.time() - start_time2
    # canny_outline_pixels = get_outline_pixels(edge)

    outline_pixels_v3, outline_normals_v3, index_array = get_outline_and_normal(roi, 150, 540, 7)

    start_time3 = time.time()
    #grasp_path_corners, contact_seeds, \
    contact_region = find_contact_region(outline_pixels_v3, index_array, [320, 651], 10)
    time3 = time.time() - start_time3

    #print("v1 outline and normal extraction used", time1, "seconds")
    #print("v2 outline and normal extraction used", time2, "seconds")
    print("v3 outline and normal extraction used", time3, "seconds")

    print(outline_pixels.shape)
    print(normals.shape)
    print(outline_pixels_new.shape)
    print(outline_normals.shape)
    print(outline_pixels_v3.shape)
    print(outline_normals_v3.shape)
    #print(index_array)
    print(index_array.shape)
    print(outline_pixels_v3[0], outline_pixels_v3[-1])


    #print(grasp_path_corners)
    #print('contact seeds', contact_seeds)
    print('r1', outline_pixels_v3[0, 0])
    print('contact region size', contact_region.shape)
    #x = np.array([[grasp_path_corners[1, 0], grasp_path_corners[1, 1]], [grasp_path_corners[1, 0], grasp_path_corners[1, 2]], [grasp_path_corners[1, 3], grasp_path_corners[1, 1]], [grasp_path_corners[1, 3], grasp_path_corners[1, 2]]])
    #y = np.array([[grasp_path_corners[0, 0], grasp_path_corners[0, 1]], [grasp_path_corners[0, 0], grasp_path_corners[0, 2]], [grasp_path_corners[0, 3], grasp_path_corners[0, 1]], [grasp_path_corners[0, 3], grasp_path_corners[0, 2]]])

    # print(canny_outline_pixels.shape)

    normal_vectors = normal_end(outline_pixels_v3, outline_normals_v3)
    # np.save('outline_row_index.txt', canny_outline_pixels)
    # with open('outline_row_index.txt', 'w') as f:
    #     for line in canny_outline_pixels:
    #         np.savetxt(f, line, fmt='%.2f')
    outline_v3 = np.zeros((800, 1280), dtype=np.uint8)
    for pixel in outline_pixels_v3:
        outline_v3[pixel[0], pixel[1]] = 1

    plt.figure(1)
    plt.imshow(mask_img * 255)
    for n_vector in normal_vectors:
        plt.plot([n_vector[1], n_vector[3]], [n_vector[0], n_vector[2]], color='g')

    plt.figure(2)
    plt.title('Outline of the object(old)')
    plt.imshow(outline)
    plt.scatter(center[1], center[0], color='r')

    plt.figure(3)
    plt.title('Outline of the object(new)')
    plt.imshow(edge)

    plt.figure(4)
    plt.title('Outline of the object(v3)')
    plt.imshow(outline_v3)

    plt.figure(5)
    plt.title('contact region test')
    plt.imshow(mask_img*255)
    # for i in range(4):
    #     plt.plot(x[i], y[i], color='g')
    plt.scatter(651, 320, color='r')
    for pixel in contact_region:
        if pixel[1] == 0:
            plt.scatter(outline_pixels_v3[pixel[0]][1], outline_pixels_v3[pixel[0]][0], color='y')
        else:
            plt.scatter(outline_pixels_v3[pixel[0]][1], outline_pixels_v3[pixel[0]][0], color='b')
    # plt.scatter(outline_pixels_v3[index_array[contact_seeds[0, 0]][0]+contact_seeds[0, 1]][1], outline_pixels_v3[index_array[contact_seeds[0, 0]][0]+contact_seeds[0, 1]][0], color='b')
    # plt.scatter(outline_pixels_v3[index_array[contact_seeds[1, 0]][0]+contact_seeds[1, 1]][1], outline_pixels_v3[index_array[contact_seeds[1, 0]][0]+contact_seeds[1, 1]][0], color='y')

    plt.show()


def main():
    path = os.path.dirname(os.getcwd())
    mask_image_name = 'spoon3_mask.png'
    image_name = 'spoon3.png'
    mask_img = cv2.imread(os.path.join(path, 'pictures', mask_image_name))
    print(mask_img.shape)
    r_crop = 150
    c_crop = 540
    roi = mask_img[r_crop:570, c_crop:740, 0]
    #outline_pixels_v3, outline_normals_v3, index_array = get_outline_and_normal(mask_img[:, :, 0], 0, 0, 7)
    outline_pixels_v3, outline_normals_v3, index_array = get_outline_and_normal(roi, r_crop, c_crop, 7)
    # calculate the object skeleton///////////////
    mask_pixels = []
    r, c = roi.shape
    for i in range(r):
        for j in range(c):
            if roi[i, j] == 1:
                mask_pixels.append([i + 150, j + 540])
    object_skeleton = zhangsuen(mask_pixels, mask_img[:, :, 0])
    object_skeleton = np.array(object_skeleton)
    plt.figure()
    plt.imshow(mask_img*255)
    x = object_skeleton[:, 1]
    y = object_skeleton[:, 0]
    plt.scatter(x, y, color='r', marker='x')
    plt.show()
    # ----------------------------///////////////
    start_time3 = time.time()
    trajectory = find_gripper_trajectory(outline_pixels_v3, index_array, outline_normals_v3, object_skeleton, [314, 659, 0, -1, -1, -1, -1])
    time3 = time.time() - start_time3
    print('find trajectory used', time3, 'seconds')
    print(trajectory)

    img2 = cv2.imread(os.path.join(path, 'pictures', image_name))
    save_path_image(trajectory, img2, r'D:\Temp')


if __name__ == '__main__':
    # test1()
    main()
    # uniform_random_points(50, 12)
