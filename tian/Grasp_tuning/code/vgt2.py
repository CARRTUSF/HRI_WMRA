import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from visulize_trajectory import save_path_image, save_loss_image, get_corners
from image_processing_visualization import visualize_all
import time
from vision4grasp_tuning import find_pixel_normals, get_outline


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
    grasp_path_corners = get_corners(center, theta, 19, 130)
    r_max = int(round(np.amax(grasp_path_corners[0, :]))) - outline_pixels[0, 0]
    r_min = int(round(np.amin(grasp_path_corners[0, :]))) - outline_pixels[0, 0]

    contact_region = np.ndarray((0, 2), dtype=np.uint8)
    contact_seeds_found = False
    contact_seeds = np.array([[-1, -1], [-1, -1]])
    seeding_sample_size = 10

    search_iteration = 0
    search_indicator = [0, 0, 0, 0]
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
    while True:
        search_iteration += 1
        left_up = contact_seeds[0, 0] + search_iteration
        left_down = contact_seeds[0, 0] - search_iteration
        right_up = contact_seeds[1, 0] + search_iteration
        right_down = contact_seeds[1, 0] - search_iteration
        if not search_indicator[0] and left_up < 335:
            stopping_row = True
            p_i1 = index_array[left_up]
            p_a1 = outline_pixels[p_i1[0]:p_i1[1]+1]
            colavg1 = np.sum(p_a1[:, 1])/len(p_a1)
            for i1 in range(len(p_a1)):
                if p_a1[i1, 1] < colavg1:
                    # print('1--', p_a1[i1])
                    if inside_grasp_path(center, theta, p_a1[i1]):
                        stopping_row = False
                        contact_region = np.append(contact_region, [[p_i1[0] + i1, 0]], axis=0)
            if stopping_row:
                search_indicator[0] = 1

        if not search_indicator[1] and left_down >= 0:
            stopping_row = True
            p_i1 = index_array[left_down]
            p_a1 = outline_pixels[p_i1[0]:p_i1[1]+1]
            colavg1 = np.sum(p_a1[:, 1])/len(p_a1)
            for i1 in range(len(p_a1)):
                if p_a1[i1, 1] < colavg1:
                    # print('2--', p_a1[i1])
                    if inside_grasp_path(center, theta, p_a1[i1]):
                        stopping_row = False
                        contact_region = np.append(contact_region, [[p_i1[0] + i1, 0]], axis=0)
            if stopping_row:
                search_indicator[1] = 1

        if not search_indicator[2] and right_up < 335:
            stopping_row = True
            p_i1 = index_array[right_up]
            p_a1 = outline_pixels[p_i1[0]:p_i1[1]+1]
            colavg1 = np.sum(p_a1[:, 1])/len(p_a1)
            for i1 in range(len(p_a1)):
                if p_a1[i1, 1] > colavg1:
                    # print('3--', p_a1[i1])
                    if inside_grasp_path(center, theta, p_a1[i1]):
                        stopping_row = False
                        contact_region = np.append(contact_region, [[p_i1[0] + i1, 1]], axis=0)
            if stopping_row:
                search_indicator[2] = 1

        if not search_indicator[3] and right_down >= 0:
            stopping_row = True
            p_i1 = index_array[right_down]
            p_a1 = outline_pixels[p_i1[0]:p_i1[1]+1]
            colavg1 = np.sum(p_a1[:, 1]) / len(p_a1)
            for i1 in range(len(p_a1)):
                if p_a1[i1, 1] > colavg1:
                    if inside_grasp_path(center, theta, p_a1[i1]):
                        stopping_row = False
                        contact_region = np.append(contact_region, [[p_i1[0] + i1, 1]], axis=0)
            if stopping_row:
                search_indicator[3] = 1
        if search_indicator[0]*search_indicator[1]*search_indicator[2]*search_indicator[3] == 1:
            break

    return grasp_path_corners, contact_seeds, contact_region


def inside_grasp_path(center, theta, pixel):
    min_y = center[0] - 19 / np.cos(np.deg2rad(theta))
    max_y = center[0] + 19 / np.cos(np.deg2rad(theta))
    y = pixel[0] - np.tan(np.deg2rad(theta)) * (pixel[1] - center[1])
    if min_y <= y <= max_y:
        return True
    else:
        return False


def test_index_array(array, r1):
    current_row = r1
    ix1=0
    ix2=0
    current_index = -1
    index_array = []
    for item in array:
        current_index += 1
        if item[0] == current_row:
            ix2 = current_index
        else:
            index_array.append([ix1, ix2])
            ix1 = current_index
            ix2 = current_index
            current_row = item[0]
    index_array.append([ix1, ix2])
    return index_array


def test1():
    path = os.path.dirname(os.getcwd())
    mask_image_name = 'spray_mask.png'
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
    grasp_path_corners, contact_seeds, contact_region = find_contact_region(outline_pixels_v3, index_array, [280, 651], 30)
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


    print(grasp_path_corners)
    print('contact seeds', contact_seeds)
    print('r1', outline_pixels_v3[0, 0])
    print('contact region size', contact_region.shape)
    x = np.array([[grasp_path_corners[1, 0], grasp_path_corners[1, 1]], [grasp_path_corners[1, 0], grasp_path_corners[1, 2]], [grasp_path_corners[1, 3], grasp_path_corners[1, 1]], [grasp_path_corners[1, 3], grasp_path_corners[1, 2]]])
    y = np.array([[grasp_path_corners[0, 0], grasp_path_corners[0, 1]], [grasp_path_corners[0, 0], grasp_path_corners[0, 2]], [grasp_path_corners[0, 3], grasp_path_corners[0, 1]], [grasp_path_corners[0, 3], grasp_path_corners[0, 2]]])

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
    for i in range(4):
        plt.plot(x[i], y[i], color='g')
    plt.scatter(651, 280, color='r')
    for pixel in contact_region:
        if pixel[1] == 0:
            plt.scatter(outline_pixels_v3[pixel[0]][1], outline_pixels_v3[pixel[0]][0], color='y')
        else:
            plt.scatter(outline_pixels_v3[pixel[0]][1], outline_pixels_v3[pixel[0]][0], color='b')
    # plt.scatter(outline_pixels_v3[index_array[contact_seeds[0, 0]][0]+contact_seeds[0, 1]][1], outline_pixels_v3[index_array[contact_seeds[0, 0]][0]+contact_seeds[0, 1]][0], color='b')
    # plt.scatter(outline_pixels_v3[index_array[contact_seeds[1, 0]][0]+contact_seeds[1, 1]][1], outline_pixels_v3[index_array[contact_seeds[1, 0]][0]+contact_seeds[1, 1]][0], color='y')

    plt.show()

    plt.show()


def test2():
    array = np.array([[23,34],[23,40],[23,50],[23,51],[24,4],[24,14],[24,24],[25,5],[25,15],[25,25],[26,26]])
    index_array = test_index_array(array,23)
    print(index_array)


if __name__ == '__main__':
    test1()