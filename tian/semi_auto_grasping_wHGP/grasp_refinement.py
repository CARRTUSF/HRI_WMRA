import numpy as np
import cv2
from tt_grasp_evaluation import evaluate_grasp


def get_corners(center, theta, ghh, ghw):
    theta = np.deg2rad(theta)
    corner_vectors_original = np.array(
        [[-ghh, ghh, ghh, -ghh], [-ghw, -ghw, ghw, ghw], [1, 1, 1, 1]])
    transformation = np.array(
        [[np.cos(theta), np.sin(theta), center[0]], [-np.sin(theta), np.cos(theta), center[1]], [0, 0, 1]])
    new_corners = np.matmul(transformation, corner_vectors_original)
    return new_corners


def plot_grasp_path(center, theta, ghh, ghw, img):
    new_corners = get_corners(center, theta, ghh, ghw)
    for k in range(4):
        if k != 3:
            cv2.line(img, (int(round(new_corners[1][k])), int(round(new_corners[0][k]))),
                     (int(round(new_corners[1][k + 1])), int(round(new_corners[0][k + 1]))), (0, 0, 255))
        else:
            cv2.line(img, (int(round(new_corners[1][3])), int(round(new_corners[0][3]))),
                     (int(round(new_corners[1][0])), int(round(new_corners[0][0]))), (0, 0, 255))
    cv2.circle(img, (center[1], center[0]), 3, (0, 255, 0), -1)


if __name__ == '__main__':
    init_grasp_center = [96, 267]
    ang = -0.4076
    width = 88.2

    color_img = cv2.imread('0_color.jpg')
    mask_img = cv2.imread('0_mask.jpg')
    print(mask_img.dtype)
    mask_0 = np.array(mask_img[:, :, 0] / 255, dtype=np.uint8)

    cv2.circle(
        color_img, (init_grasp_center[1], init_grasp_center[0]), 4, (0, 0, 255), -1)
    dy = int(round(np.sin(ang) * width / 2))
    dx = int(round(np.cos(ang) * width / 2))
    cv2.line(color_img, (int(round(init_grasp_center[1] - dx)), int(round(init_grasp_center[0] - dy))),
             (int(round(init_grasp_center[1] + dx)), int(round(init_grasp_center[0] + dy))), (255, 0, 0), 2)
    cv2.imshow('results', np.hstack((color_img, mask_img)))
    cv2.waitKey(0)
    print(mask_0.dtype)
    print(np.amax(mask_0))
    print(np.amin(mask_0))
    print(np.median(mask_0))

    grasp_found = False
    desired_grasp_score = 0.95
    time_out = 0
    hw = 44
    hh = 10
    while not grasp_found and time_out < 2000:

        gx = np.random.randint(-30, 30) + init_grasp_center[1]
        gy = np.random.randint(-30, 30) + init_grasp_center[0]
        a = 0
        search_show = np.copy(color_img)
        cv2.circle(search_show, (gx, gy), 4, (0, 0, 255), -1)
        cv2.imshow('grasp refinement', search_show)
        cv2.waitKey(1)
        while a < 10:
            time_out += 1
            a += 1
            ang = np.random.randint(-90, 90)
            grasp = [gy, gx, ang, hh, hw]
            # print grasp
            score, features = evaluate_grasp(grasp, mask_0)
            print(score)
            print(features)
            if score > desired_grasp_score:
                grasp_found = True
                grasp_rect = [gy, gx, ang]
                plot_grasp_path([gy, gx], ang, hh, hw, search_show)
                cv2.imshow('grasp refinement', search_show)
                cv2.waitKey(0)
                break
