import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from visulize_trajectory import save_path_image, save_loss_image, get_corners
from helpers import extract_contact_region, high_level_grasp_feature, grasp_quality_score_v2, cv_plot_contact_region
import time

next_pose1 = []


def find_next_gripper_pose(current_pos, angle_search_space, mask_image, gripper_hh, gripper_hw):
    r = 8
    threshold = 0.005
    max_score = current_pos[3]
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
            contact_rl, contact_rr = extract_contact_region(next_row, next_col, next_theta, gripper_hh, gripper_hw,
                                                            mask_image[:, :, 0])
            cli, trans, rotation, slippage, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr,
                                                                                          next_theta, gripper_hh,
                                                                                          gripper_hw)
            score = grasp_quality_score_v2(cli, trans, rotation, slippage, offs, gripper_hh)
            score = grasp_quality_score_v2(cli, trans, rotation, slippage, offs, gripper_hh)
            # print('grasp score', scores[0][0])
            if score - max_score > threshold:
                # print('grasp score1', max_score)
                max_score = score
                # print('grasp score2', max_score)
                next_pos = [next_row, next_col, next_theta, max_score]
                ####### plot contact region for debuging#########
                # plot_img = np.copy(mask_image*255)
                # cv_plot_contact_region(contact_rl, plot_img, True, lcids)
                # cv_plot_contact_region(contact_rr, plot_img, True, rcids)
                # img_name = str(time.time()) + 'contact.png'
                # save_dir = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset\refined_results'
                # save_file = os.path.join(save_dir, img_name)
                # cv2.imwrite(save_file, plot_img)
                #################################################
            # print("one search used", time.time()-s_time, "seconds")
    return next_pos


def find_gripper_trajectory(mask_image, current_pos, gripper_hh, gripper_hw):
    # current_pos = [r, c, ang]
    iterations = 0

    contact_rl, contact_rr = extract_contact_region(current_pos[0], current_pos[1], current_pos[2], gripper_hh,
                                                    gripper_hw, mask_image[:, :, 0])
    cli, trans, rotation, slippage, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr,
                                                                                  current_pos[2], gripper_hh,
                                                                                  gripper_hw)
    score = grasp_quality_score_v2(cli, trans, rotation, slippage, offs, gripper_hh)
    current_pos[3] = score
    trajectory = [current_pos]
    # angle_search_space = np.arange(-90, 96, 6)
    # angle_search_space = np.append([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], angle_search_space)
    angle_search_space = np.arange(-30, 30, 5)
    angle_search_space = np.append([-4, -3, -2, -1, 1, 2, 3, 4], angle_search_space)
    while iterations < 20:
        start_time = time.time()
        iterations += 1
        next_pos = find_next_gripper_pose(current_pos, angle_search_space, mask_image, gripper_hh, gripper_hw)
        if current_pos == next_pos:
            break
        current_pos = next_pos
        trajectory.append(current_pos)
        print("iteration", iterations, "used", time.time() - start_time, "seconds")
    trajectory = np.array(trajectory)
    print('final score:', trajectory[-1, 3])
    return trajectory


def refine_gripper_pose(mask_image, current_pos, gripper_hh, gripper_hw):
    # current_pos = [r, c, ang]
    iterations = 0

    contact_rl, contact_rr = extract_contact_region(current_pos[0], current_pos[1], current_pos[2], gripper_hh,
                                                    gripper_hw, mask_image[:, :, 0])
    cli, trans, rotation, slippage, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr,
                                                                                  current_pos[2], gripper_hh,
                                                                                  gripper_hw)
    score = grasp_quality_score_v2(cli, trans, rotation, slippage, offs, gripper_hh)
    current_pos[3] = score
    trajectory = [current_pos]
    # angle_search_space = np.arange(-90, 96, 6)
    # angle_search_space = np.append([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], angle_search_space)
    angle_search_space = np.arange(-30, 30, 5)
    angle_search_space = np.append([-4, -3, -2, -1, 1, 2, 3, 4], angle_search_space)
    while iterations < 20:
        iterations += 1
        next_pos = find_next_gripper_pose(current_pos, angle_search_space, mask_image, gripper_hh, gripper_hw)
        if current_pos == next_pos:
            break
        current_pos = next_pos
        trajectory.append(current_pos)
    trajectory = np.array(trajectory)
    # print('final score:', trajectory[-1, 3])
    return trajectory


def main():
    path = os.path.dirname(os.getcwd())
    mask_image_name = 'spoon3_mask.png'
    image_name = 'spoon3.png'
    object_name = ''.join(list(image_name)[:-4])
    mask_img = cv2.imread(os.path.join(path, 'pictures', mask_image_name))
    print(mask_img.shape)
    # cv2.imshow('mask', mask_img*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    start_time3 = time.time()
    trajectory = find_gripper_trajectory(mask_img, [314, 659, 0, -1], 19, 130)
    time3 = time.time() - start_time3
    print('find trajectory used', time3, 'seconds')
    print(trajectory)

    img2 = cv2.imread(os.path.join(path, 'pictures', image_name))
    save_path_image(trajectory, img2, r'D:\Temp\trj_' + object_name)


if __name__ == '__main__':
    # test1()
    main()
    # uniform_random_points(50, 12)
