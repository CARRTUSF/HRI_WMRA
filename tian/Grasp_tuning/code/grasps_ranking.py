import numpy as np
import cv2
from helpers import plot_grasp_rect
import os
import time


def compare(g1, g2, pre_scores, img):
    img1 = np.copy(img)
    img2 = np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = f'{pre_scores[0], pre_scores[1], pre_scores[2], pre_scores[3]}'
    cv2.putText(img1, text1, (50, 100), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
    plot_grasp_rect(g1, 19, 130, img1)
    plot_grasp_rect(g2, 19, 130, img2)
    img3 = cv2.hconcat([img1, img2])
    img3_s = cv2.resize(img3, (1280, 400))
    cv2.imshow('choose the better grasp', img3_s)
    usr_input = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if usr_input == ord('1'):
        return 1
    elif usr_input == ord('2'):
        return 2
    elif usr_input == ord('0'):
        return 0
    elif usr_input == ord('q'):
        return 3


def find_grasp_ranking(di, img):
    global results
    gi = di[-3:]
    upgrade = False
    place = -1
    score = -1
    if results.shape[0] == 0:
        place = 0
        score = 0
    else:
        k1 = 0
        k2 = results.shape[0] - 1
        cg = [results[k1, -3], results[k1, -2], results[k1, -1]]
        r = compare(gi, cg, di[:4], img)
        if r == 0:  # equal to the current lowest
            place = 1
            score = 0
        elif r == 2:  # worst than the current lowest
            place = 0
            score = 0
            upgrade = True
        elif r == 1:  # better than current lowest
            cg = [results[k2, -3], results[k2, -2], results[k2, -1]]
            r = compare(gi, cg, di[:4], img)
            if r == 0:  # equal to the current best
                place = k2
                score = results[k2, 0]  # score of k2
            elif r == 1:  # better than the current best
                place = k2 + 1
                score = results[k2, 0] + 1  # score of k2 + 1
            elif r == 2:
                while True:
                    km = int((k1 + k2) / 2)
                    print('-------------', k1, k2, km)
                    if km == k1:
                        place = k1 + 1
                        if results[k1, 0] == results[k2, 0]:
                            score = results[k1, 0]
                        else:
                            score = results[k2, 0]  # score of k2
                            upgrade = True
                        break
                    else:
                        cg = [results[km, -3], results[km, -2], results[km, -1]]
                        r = compare(gi, cg, di[:4], img)
                        if r == 0:  # same as km
                            place = km
                            score = results[km, 0]  # score of km
                            break
                        elif r == 1:  # better than km
                            k1 = km
                        elif r == 2:  # worst than km
                            k2 = km
    return place, score, upgrade


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    img_rgb = 'wbt.png'
    img_mask = 'wbt_mask.png'
    # I = cv2.imread(os.path.join(path, 'pictures', img_rgb))
    Im = cv2.imread(os.path.join(path, 'pictures', img_mask))
    # Ip = get_object_pixels(Im[:, :, 0])
    # cv2.imshow('image', I)
    # key = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(key)
    # print(ord('1'))
    # print(ord('2'))
    # print(ord('0'))
    x = np.loadtxt('unranked_grasp_data.txt')
    results = np.ndarray((0, x.shape[1] + 1), dtype=np.float16)
    mask_img = Im * 255
    for i in range(205, x.shape[0]):
        # row = x[i, -3]
        # col = x[i, -2]
        # ang = x[i, -1]
        # pre_score = x[i, 0]
        # graspi = [row, col, ang]
        placei, scorei, updatei = find_grasp_ranking(x[i], mask_img)
        print(placei, scorei, updatei)
        if placei == -1:
            break
        new_grasp_data = np.concatenate([[scorei], x[i]])
        if updatei:
            results[placei:, 0] += 1
        results = np.concatenate([results[:placei, :], [new_grasp_data], results[placei:, :]])
        print(results.shape)
    np.savetxt(f'human_ranked_data.txt', results, fmt='%1.4f')
