import os
import cv2
import numpy as np
import random
from vgt2 import find_contact_region, get_outline_and_normal
from visulize_trajectory import plot_grasp_path


def get_object_pixels(mask_img):
    object_pixels = []
    r, c = mask_img.shape
    for i in range(r):
        for j in range(c):
            if mask_img[i, j] == 1:
                object_pixels.append([i, j])
    return object_pixels


def grasp_features(contact_region, outline_pixels, normals, gripper_center, theta):    # gripper_center(row,col)
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
    if contact_region.shape[0] > 60:
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
            alpha = 180
            beta = 180
        elif right_sum.all() == 0:
            beta = 180
            alpha = 180
        else:
            c_alpha = np.dot(l_normal, left_sum) / (left_sum[0]**2+left_sum[1]**2)**0.5
            c_beta = np.dot(r_normal, right_sum) / (right_sum[0]**2 + right_sum[1]**2)**0.5
            alpha = np.rad2deg(np.arccos(c_alpha))
            beta = np.rad2deg(np.arccos(c_beta))
        theta_loss = alpha + beta
        return x_loss, y_loss, theta_loss
    else:
        return -1, -1, -1


def gather_data(mask_img, obj_pt, center_id, grasp_angles):
    outline_pixels, outline_normals, index_array = get_outline_and_normal(mask_img[:, :, 0], 0, 0, 7)
    results = np.ndarray((0, 7), dtype=np.float16)
    # interrupted = False
    for i in center_id:
        # if interrupted:
        #     break
        for a in grasp_angles:
            # x = input('press q to quit...')
            # if x == 'q':
            #     interrupted = True
            #     break
            contact = find_contact_region(outline_pixels, index_array, obj_pt[i], a)
            l1, l2, l3 = grasp_features(contact, outline_pixels, outline_normals, obj_pt[i], a)
            data = [0, obj_pt[i][0], obj_pt[i][1], a, l1, l2, l3]  # current grasp data
            if results.size == 0:
                new_results = np.append(results, [data], axis=0)
                # add first one
            else:
                left = 0
                right = len(results) - 1

                ci = results[left]
                img1 = np.copy(mask_img*255)
                img2 = np.copy(mask_img*255)
                plot_grasp_path(obj_pt[i], a, 19, 130, img1)
                plot_grasp_path([int(round(ci[1])), int(round(ci[2]))], ci[3], 19, 130, img2)
                img3 = cv2.hconcat([img1, img2])
                img3_s = cv2.resize(img3, (1280, 400))
                cv2.imshow('choose the better grasp', img3_s)
                print(f'compare with {left}')
                usr_input = cv2.waitKeyEx(0)
                cv2.destroyAllWindows()
                if usr_input == 2424832:
                    # case 3
                    print('left side is better')
                    cv2.destroyAllWindows()
                    ci = results[right]
                    img1 = np.copy(mask_img*255)
                    img2 = np.copy(mask_img*255)
                    plot_grasp_path(obj_pt[i], a, 19, 130, img1)
                    plot_grasp_path([int(round(ci[1])), int(round(ci[2]))], ci[3], 19, 130, img2)
                    img3 = cv2.hconcat([img1, img2])
                    img3_s = cv2.resize(img3, (1280, 400))
                    cv2.imshow('choose the better grasp', img3_s)
                    print(f'compare with {right}')
                    usr_input = cv2.waitKeyEx(0)
                    cv2.destroyAllWindows()
                    if usr_input == 2555904:
                        # case 31
                        print('right side is better')
                        while True:
                            mid = int((left + right) / 2)
                            if mid == left:
                                print(f'left {left}, right {right}')
                                data[0] = results[right, 0]
                                results[right:, 0] += 1
                                new_results = np.concatenate([results[:right], [data], results[right:]])
                                break
                            else:
                                ci = results[mid]
                                img1 = np.copy(mask_img*255)
                                img2 = np.copy(mask_img*255)
                                plot_grasp_path(obj_pt[i], a, 19, 130, img1)
                                plot_grasp_path([int(round(ci[1])), int(round(ci[2]))], ci[3], 19, 130, img2)
                                img3 = cv2.hconcat([img1, img2])
                                img3_s = cv2.resize(img3, (1280, 400))
                                cv2.imshow('choose the better grasp', img3_s)
                                print(f'compare with {mid}')
                                usr_input = cv2.waitKeyEx(0)
                                cv2.destroyAllWindows()
                                if usr_input == 2424832:
                                    print('left side is better')
                                    left = mid
                                elif usr_input == 2555904:
                                    print('right side is better')
                                    right = mid
                                else:
                                    print('same quality')
                                    data[0] = ci[0]
                                    better_side = mid
                                    for n in range(mid, len(results)):
                                        if results[n, 0] > ci[0]:
                                            better_side = n
                                            break
                                    results[better_side:, 0] += 1
                                    new_results = np.concatenate([results[:mid], [data], results[mid:]])
                                    break

                    elif usr_input == 2424832:
                        # case 32
                        print('current pose is the best so far')
                        data[0] = ci[0] + 1     # update grasp quality
                        new_results = np.concatenate([results, [data]])  # add current grasp to the array
                    else:
                        # case 33
                        print('current pose is equally good as the best one')
                        data[0] = ci[0]
                        new_results = np.concatenate([results, [data]])

                elif usr_input == 2555904:
                    print('right side is better')
                    # case 1
                    results[:, 0] += 1  # update grasp quality of all grasps that are better that current grasp
                    new_results = np.concatenate([[data], results])     # add current grasp to the array
                else:
                    # case 2
                    print('same grasp quality')
                    new_results = np.concatenate([[data], results])

            results = new_results
            print(results.shape)
            np.savetxt('grasp_feature_study.txt', results, fmt='%1.4f')
    return results


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    img_rgb = 'spoon3.png'
    img_mask = 'spoon3_mask.png'
    I = cv2.imread(os.path.join(path, 'pictures', img_rgb))
    Im = cv2.imread(os.path.join(path, 'pictures', img_mask))
    Ip = get_object_pixels(Im[:, :, 0])
    while True:
        sample_indices = random.sample(range(len(Ip)), 20)
        sample_check = np.copy(Im)*255
        for ii in sample_indices:
            cv2.circle(sample_check, (Ip[ii][1], Ip[ii][0]), 2, (0, 0, 255))
        cv2.imshow('sample check', sample_check)
        print('press k if the sample is good')
        xx = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if xx == ord('k') or xx == ord('K'):
            break
    sample_angles = np.arange(-90, 90, 10)

    gf_data = gather_data(Im, Ip, sample_indices, sample_angles)
    np.savetxt('grasp_feature_study.txt', gf_data, fmt='%1.4f')

    # arr = np.arange(9).reshape((3, 3))
    # new_arr = np.concatenate([arr[:2], [[8,3,3]], arr[2:]])
    # print(new_arr)

    # path = os.path.dirname(os.getcwd())
    # img_rgb = 'spoon3.png'
    # img_mask = 'spoon3_mask.png'
    # I = cv2.imread(os.path.join(path, 'pictures', img_rgb))
    # print('press a key')
    # cv2.imshow('test', I)
    # c = cv2.waitKeyEx(0)
    # print('you pressed', c)
    # if c == 2621440:
    #     print('you pressed down')
    # if c == 2424832:
    #     print('you pressed left')
    # if c == 2555904:
    #     print('you pressed '
    #           'right')