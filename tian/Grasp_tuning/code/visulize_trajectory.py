import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


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
    cv2.circle(img, (center[1], center[0]), 3, (0, 255, 0), -1)


def find_new_corner_vectors(corner_vectors, angle):
    r = (130 ** 2 + 18 ** 2) ** 0.5
    new_corner_vectors = []
    for vector in corner_vectors:
        ca = vector[0] / r
        sa = vector[1] / r
        new_vx = (np.cos(np.deg2rad(angle)) * ca - np.sin(np.deg2rad(angle)) * sa) * r
        new_vy = (np.sin(np.deg2rad(angle)) * ca + np.cos(np.deg2rad(angle)) * sa) * r
        new_corner_vectors.append([new_vx, new_vy])
    new_corner_vectors = np.array(new_corner_vectors)
    return new_corner_vectors


def save_path_image(trajectory, img, path):
    corner_vectors = np.array([[-130, -18], [130, -18], [130, 18], [-130, 18]])
    new_corner_vectors = corner_vectors
    for i in range(trajectory.shape[0]):
        img_copy = np.copy(img)
        if i != 0:
            new_corner_vectors = find_new_corner_vectors(corner_vectors, trajectory[i][2])
        center = [int(round(trajectory[i][1])), int(round(trajectory[i][0]))]
        new_corners = new_corner_vectors + np.array([center, center, center, center])
        for k in range(4):
            if k != 3:
                cv2.line(img_copy, (int(round(new_corners[k][0])), int(round(new_corners[k][1]))),
                         (int(round(new_corners[k + 1][0])), int(round(new_corners[k + 1][1]))), (0, 0, 255))
            else:
                cv2.line(img_copy, (int(round(new_corners[3][0])), int(round(new_corners[3][1]))),
                         (int(round(new_corners[0][0])), int(round(new_corners[0][1]))), (0, 0, 255))
        cv2.circle(img, (center[0], center[1]), 3, (0, 255, 0), -1)
        img_name = path + "\\" + "%d.png" % i
        cv2.imwrite(img_name, img_copy)


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
    new_corners = get_corners([8, 12], 35, 4, 10)
    print(new_corners)
    print(np.amax(new_corners[0,:]))
    print(np.amin(new_corners[0,:]))
    # path = os.path.dirname(os.getcwd())
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
