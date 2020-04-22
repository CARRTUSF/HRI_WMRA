import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def get_corners(center, theta, h, w):
    """

    :param center:
    :param theta:
    :param h:
    :param w:
    :return: [
    """
    theta = np.deg2rad(theta)
    corner_vectors_original = np.array([[-h, h, h, -h], [-w, -w, w, w], [1, 1, 1, 1]])
    transformation = np.array(
        [[np.cos(theta), np.sin(theta), center[0]], [-np.sin(theta), np.cos(theta), center[1]], [0, 0, 1]])
    new_corners = np.matmul(transformation, corner_vectors_original)
    return new_corners


def plot_grasp_path(center, theta, h, w, img):
    new_corners = get_corners(center, theta, h, w)
    for k in range(4):
        if k != 3:
            cv2.line(img, (int(round(new_corners[1][k])), int(round(new_corners[0][k]))),
                     (int(round(new_corners[1][k + 1])), int(round(new_corners[0][k + 1]))), (0, 0, 255))
        else:
            cv2.line(img, (int(round(new_corners[1][3])), int(round(new_corners[0][3]))),
                     (int(round(new_corners[1][0])), int(round(new_corners[0][0]))), (0, 0, 255))
    cv2.circle(img, (center[1], center[0]), 3, (0, 50, 255), -1)
    # for i in range(1, h):
    #     tan_theta = np.tan(np.deg2rad(theta))
    #     dc = (1 + tan_theta ** 2) ** 0.5
    #     ur_c = int(round(center[0] - i / dc))
    #     uc_c = int(round(center[1] + i * tan_theta / dc))
    #
    #     dr_c = int(round(center[0] + i / dc))
    #     dc_c = int(round(center[1] - i * tan_theta / dc))
    #     cv2.circle(img, (uc_c, ur_c), 2, (255, 0, 0), -1)
    #     cv2.circle(img, (dc_c, dr_c), 2, (0, 0, 255), -1)


def find_new_corner_vectors(corner_vectors, angle):
    # r = (ghw ** 2 + ghh ** 2) ** 0.5
    # new_corner_vectors = []
    # for vector in corner_vectors:
    #     ca = vector[0] / r
    #     sa = vector[1] / r
    #     new_vx = (np.cos(np.deg2rad(angle)) * ca - np.sin(np.deg2rad(angle)) * sa) * r
    #     new_vy = (np.sin(np.deg2rad(angle)) * ca + np.cos(np.deg2rad(angle)) * sa) * r
    #     new_corner_vectors.append([new_vx, new_vy])
    # new_corner_vectors = np.array(new_corner_vectors)
    new_corner_vectors = np.ndarray((0, 2), dtype=np.float16)
    cos = np.cos(np.deg2rad(angle))
    sin = np.sin(np.deg2rad(angle))
    r = np.array([[cos, -sin], [sin, cos]])
    for vector in corner_vectors:
        vector = np.array([[vector[0]], [vector[1]]])
        new_vector = np.matmul(r, vector)
        new_corner_vectors = np.append(new_corner_vectors, new_vector.reshape(1, 2), axis=0)
    return np.rint(new_corner_vectors).astype(int)


def save_path_image(trajectory, img, path, ghh=18, ghw=130, save_rect=False, init_score=-2):
    """

    :param trajectory:
    :param ghh: gripper rectangle half height
    :param ghw: gripper rectangle half width
    :param img:
    :param path: path to save the trajectory images
    :param save_rect:
    :param init_score:
    :return:
    """
    corner_vectors = np.array([[-ghw, -ghh], [ghw, -ghh], [ghw, ghh], [-ghw, ghh]])
    new_corners = np.array([[-1]])
    for i in range(trajectory.shape[0]):
        img_copy = np.copy(img)
        new_corner_vectors = find_new_corner_vectors(corner_vectors, trajectory[i][2])
        center = [int(round(trajectory[i][1])), int(round(trajectory[i][0]))]
        new_corners = new_corner_vectors + np.array([center, center, center, center])
        for k in range(4):
            line_width = 2
            line_color = (255, 0, 0)
            if k != 3:
                if k == 0 or k == 2:
                    line_width = 1
                    line_color = (255, 255, 0)
                cv2.line(img_copy, (new_corners[k][0], new_corners[k][1]),
                         (new_corners[k + 1][0], new_corners[k + 1][1]), line_color, line_width)
            else:
                cv2.line(img_copy, (new_corners[3][0], new_corners[3][1]),
                         (new_corners[0][0], new_corners[0][1]), line_color, line_width)
        cv2.circle(img_copy, (center[0], center[1]), 2, (0, 255, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        score_text = f'{trajectory[i][3]:.4f}'
        if init_score != -2:
            init_score = f'{init_score:.4f}'
            cv2.putText(img_copy, init_score, (50, 50), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img_copy, score_text, (50, 80), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        img_name = path + "%d.png" % i
        cv2.imwrite(img_name, img_copy)
    if save_rect:
        file_name = path + ".txt"
        if new_corners[0, 0] != -1:
            # print(new_corners)
            new_corners2 = np.ndarray((4, 2), dtype=int)
            new_corners2[:, 0] = new_corners[:, 1]
            new_corners2[:, 1] = new_corners[:, 0]
            # print(new_corners2)
            temp = np.copy(new_corners2[1, :])
            new_corners2[1, :] = new_corners2[3, :]
            new_corners2[3, :] = temp
            # print(new_corners2)
        np.savetxt(file_name, new_corners2, fmt='%u')


def save_loss_image(loss):
    for i in range(loss.shape[0]):
        x = np.arange(0, i + 1)
        y = loss[:i + 1]
        # y_max = np.amax(y)
        plt.figure()
        plt.xlim(0, 15)
        plt.xticks(range(0, 15))
        plt.plot(x, y)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        fig_name = "00%d.png" % i
        plt.savefig(fig_name)
        plt.clf()


def main():
    trajectory = [[3.14000000e+02, 6.59000000e+02, 0.00000000e+00, 6.20956106e-01],
                  [3.21122729e+02, 6.59720081e+02, 2.50000000e+01, 8.51581774e-01],
                  [3.25724929e+02, 6.64915593e+02, 2.80000000e+01, 9.42412786e-01],
                  [3.25415950e+02, 6.72738615e+02, 3.00000000e+01, 9.71184605e-01]]
    trajectory = np.array(trajectory)
    path = os.path.dirname(os.getcwd())
    image_name = 'spoon3_mask.png'
    img2 = cv2.imread(os.path.join(path, 'pictures', image_name))
    save_path_image(trajectory, img2 * 255, r'D:\Temp')
    # image_name = 'lalala.png'
    # img = cv2.imread(os.path.join(path, 'pictures', image_name))

    # corner_vectors = np.array([[-130, -18], [130, -18], [130, 18], [-130, 18]])
    #
    # new_corner_vectors = find_new_corner_vectors(corner_vectors, -5)
    # print(corner_vectors.shape)
    # print(new_corner_vectors)
    # center = [634, 341]
    # new_corners = new_corner_vectors + np.array([center, center, center, center])
    # print(new_corners)
    # for i in range(4):
    #     print(new_corners[i])
    #     if i != 3:
    #         cv2.line(img, (int(round(new_corners[i][0])), int(round(new_corners[i][1]))), (int(round(new_corners[i+1][0])), int(round(new_corners[i+1][1]))), (0, 0, 255))
    #     else:
    #         cv2.line(img, (int(round(new_corners[3][0])), int(round(new_corners[3][1]))), (int(round(new_corners[0][0])), int(round(new_corners[0][1]))), (0, 0, 255))
    # cv2.circle(img,(659, 314),3,(0,255,0),-1)
    # # cv2.circle(img, (center[0], center[1]), 3, (0, 255, 0), -1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite("00.png", img)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
