import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from visulize_trajectory import save_path_image, save_loss_image
from image_processing_visualization import visualize_all
import time
from vision4grasp_tuning import find_pixel_normals


def get_outline(obj_mask):
    """

    :param obj_mask: image mask
    :return:
        outline: binary image of the image outline
        outline_pixel: pixel indices of the outline
            nx[row, col, side] n is the number of outline points, side indicates left or right outline point
        center: center of the outline [row, col]

    """
    r, c = obj_mask.shape
    outline_pixels = np.ndarray((0, 3), dtype=np.uint8)
    outline = np.zeros((r, c), np.uint8)
    row_sum = 0
    pixel_count = 0
    col_sum = 0
    # get outline
    for i in range(r):
        left_index = c
        for j in range(c):
            if obj_mask[i, j] == 1:
                outline_pixels = np.append(outline_pixels, [[i, j, 0]], axis=0)
                outline[i, j] = 1
                row_sum += i
                col_sum += j
                pixel_count += 1
                left_index = j
                break
        if left_index >= c-1:
            continue
        else:
            for k in range(c):
                if obj_mask[i, c - k - 1] == 1:
                    outline_pixels = np.append(outline_pixels, [[i, c - k - 1, 1]], axis=0)
                    outline[i, c - k - 1] = 1
                    row_sum += i
                    col_sum += c - k - 1
                    pixel_count += 1
                    break
    # get outline image
    center = np.array([row_sum/pixel_count, col_sum/pixel_count])
    return outline, outline_pixels, center


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


def find_contact_region(outline_image, center, theta):
    path_vr0 = np.arange(-19, 20)   # 39 rows
    path_vc0 = np.arange(-130, 131)     # 261 cols
    pcv0 = np.repeat(path_vc0, 39).reshape((1, -1))
    prv0 = np.repeat(path_vr0.reshape((1, -1)), 261, axis=0).reshape((1, -1))
    path_v0 = np.concatenate((prv0, pcv0), axis=0)


def test():
    path = os.path.dirname(os.getcwd())
    mask_image_name = 'spoon3_mask.png'
    image_name = 'spoon3.png'
    mask_img = cv2.imread(os.path.join(path, 'pictures', mask_image_name))
    print(mask_img.shape)
    roi = mask_img[250:470, 540:740, 0]
    start_time = time.time()
    outline, outline_pixels, center = get_outline(roi)
    normals = find_pixel_normals(outline, outline_pixels, center, 7)
    print("old outline and normal extraction used", time.time()-start_time, "seconds")
    start_time = time.time()
    edge = cv2.Canny(roi*255, 100, 200)
    edge = (edge/255).astype(np.uint8)
    outline_pixels_new, outline_normals = get_outline_normal(roi, edge, 250, 540, 5)
    #canny_outline_pixels = get_outline_pixels(edge)
    print("new outline and normal extraction used", time.time() - start_time, "seconds")

    print(outline_pixels.shape)
    #print(canny_outline_pixels.shape)

    normal_vectors = normal_end(outline_pixels_new, outline_normals)
    # np.save('outline_row_index.txt', canny_outline_pixels)
    # with open('outline_row_index.txt', 'w') as f:
    #     for line in canny_outline_pixels:
    #         np.savetxt(f, line, fmt='%.2f')

    plt.figure(1)
    plt.imshow(mask_img*255)
    for n_vector in normal_vectors:
        plt.plot([n_vector[1], n_vector[3]], [n_vector[0], n_vector[2]], color='g')

    plt.figure(2)
    plt.title('Outline of the object(old)')
    plt.imshow(outline)
    plt.scatter(center[1], center[0], color='r')

    plt.figure(3)
    plt.title('Outline of the object(new)')
    plt.imshow(edge)

    plt.show()


if __name__ == '__main__':
    test()
