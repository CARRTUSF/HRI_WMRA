import numpy as np
import os
import cv2
import time
import random


def extract_silhouette_center():
    """

    :return: 1, extracted center/a random point on the object silhouette
             2, all the object points ids (id=r*w+c)
    """

    path = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'
    n = 370
    data = np.loadtxt(os.path.join(path, 'IdMapping.txt'), dtype=np.int16, delimiter=',')
    # print(data)
    for i in range(n):
        inst_n = data[i, 1]
        if inst_n >= 1000:
            mask_image_name = 'pcd' + str(inst_n) + 'mask.png'
            save_id = str(inst_n)
        else:
            mask_image_name = 'pcd0' + str(inst_n) + 'mask.png'
            save_id = '0' + str(inst_n)
        # print(os.path.join(path, mask_image_name))
        mask_img = cv2.imread(os.path.join(path, 'masks', mask_image_name))
        h = mask_img.shape[0]
        w = mask_img.shape[1]
        # t_start = time.time()
        mask_m = mask_img[:, :, 0] / 255
        row_v = np.arange(h).reshape((-1, 1))
        col_v = np.arange(w).reshape((1, -1))
        row_m = np.repeat(row_v, w, axis=1)
        col_m = np.repeat(col_v, h, axis=0)
        row_mm = mask_m * row_m
        col_mm = mask_m * col_m
        row_sum = np.sum(row_mm)
        col_sum = np.sum(col_mm)
        count = 0
        ind = []
        for j in range(h * w):
            r = int(j / 640)
            c = j % 640
            if mask_m[r, c] == 1:
                count += 1
                ind.append(j)
        center_r = int(round(row_sum / count))
        center_c = int(round(col_sum / count))
        if mask_m[center_r, center_c] != 1 and ind != []:
            pi = random.choice(ind)
            center_r = int(pi / 640)
            center_c = pi % 640
        # t_end = time.time()
        # print('find center of a ', h, 'x', w, 'object silhouette image used', t_end-t_start, 'seconds')
        save2file = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset\masks\pcd' + save_id + 'ct.txt'
        save2file2 = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset\masks\pcd' + save_id + 'aid.txt'
        np.savetxt(save2file, [center_r, center_c], fmt='%u')
        np.savetxt(save2file2, ind, fmt='%u')
        print('progress----------', i / n * 100, '%')
        # path2save = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset\masks_with_center\pcd' + save_id + 'maskc.png'
        # cv2.circle(mask_img, (center_c, center_r), 2, (0, 255, 0), -1)
        # cv2.imwrite(path2save, mask_img)


if __name__ == '__main__':
    extract_silhouette_center()
    # x = np.arange(5).reshape((-1, 1))
    # print(x.shape)
    # print(x)
    # z = np.repeat(x, 6, axis=1)
    # print(z)
    # y = np.array([[0, 1, 0, 0, 1]])
    # v = np.repeat(y.transpose(), 6, axis=1)
    # nn = v*z
    # print(v)
    # print(nn)
