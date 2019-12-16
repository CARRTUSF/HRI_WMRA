import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


def neighbours(x, y, img):
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1, y], img[x_1, y1], img[x, y1], img[x1, y1],  # P2,P3,P4,P5
            img[x1, y], img[x1, y_1], img[x, y_1], img[x_1, y_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2, P3, ... , P8, P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


def zhangsuen(mask_pixels, image):
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1  # the points to be removed (set as 0)
    skeleton_pixels = mask_pixels.copy()
    while changing1 or changing2:  # iterates until no further changes occur in the image
        new_pixels = []
        # Step 1
        changing1 = []
        # rows, columns = Image_Thinned.shape  # x for rows, y for columns
        # for x in range(1, rows - 1):  # No. of  rows
        #     for y in range(1, columns - 1):  # No. of columns
        for pixel in skeleton_pixels:
            x = pixel[0]
            y = pixel[1]
            P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
            if (Image_Thinned[x, y] == 1 and  # Condition 0: Point P1 in the object regions
                    2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and  # Condition 2: S(P1)=1
                    P2 * P4 * P6 == 0 and  # Condition 3
                    P4 * P6 * P8 == 0):  # Condition 4
                changing1.append((x, y))
            else:
                new_pixels.append([x, y])
        for x, y in changing1:
            Image_Thinned[x, y] = 0
        print(len(new_pixels))
        skeleton_pixels = new_pixels.copy()
        new_pixels = []
        # Step 2
        changing2 = []
        # for x in range(1, rows - 1):
        #     for y in range(1, columns - 1):
        for pixel in skeleton_pixels:
            x = pixel[0]
            y = pixel[1]
            P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
            if (Image_Thinned[x, y] == 1 and  # Condition 0
                    2 <= sum(n) <= 6 and  # Condition 1
                    transitions(n) == 1 and  # Condition 2
                    P2 * P4 * P8 == 0 and  # Condition 3
                    P2 * P6 * P8 == 0):  # Condition 4
                changing2.append((x, y))
            else:
                new_pixels.append([x, y])
        for x, y in changing2:
            Image_Thinned[x, y] = 0
        print(len(new_pixels))
        skeleton_pixels = new_pixels.copy()
    return Image_Thinned, skeleton_pixels


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    mask_image_name = 'spoon3_mask.png'
    image_name = 'spoon3.png'
    mask_img = cv2.imread(os.path.join(path, 'pictures', mask_image_name))
    imgb = cv2.imread(os.path.join(path, 'pictures', image_name))
    roi = mask_img[250:500, 450:750, 0]
    mask_pixels = []
    r, c = roi.shape
    for i in range(r):
        for j in range(c):
            if roi[i, j] == 1:
                mask_pixels.append([i + 250, j + 450])
    skeleton_img, skeleton_pixs = zhangsuen(mask_pixels, mask_img[:, :, 0])
    skp = np.array(skeleton_pixs)
    plt.figure()
    plt.imshow(imgb)
    x = skp[:, 1]
    y = skp[:, 0]
    plt.scatter(x, y, color='r', marker='x')
    plt.show()
