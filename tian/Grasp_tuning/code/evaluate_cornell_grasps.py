import cv2
import numpy as np
import os
from visulize_trajectory import save_path_image
from helpers import extract_contact_region, high_level_grasp_feature, grasp_quality_score_v2
from visulize_trajectory import find_new_corner_vectors
from tqdm import tqdm


def four_pts2center_ang(fourpoints, rc_switch=False, order_shift=0):
    for shift_i in range(order_shift):
        temp = np.copy(fourpoints[0, :])
        fourpoints[:-1] = fourpoints[1:]
        fourpoints[-1] = temp
    if rc_switch:
        rows = fourpoints[:, 1]
        cols = fourpoints[:, 0]
    else:
        rows = fourpoints[:, 0]
        cols = fourpoints[:, 1]
    rec_center_row = int(round(np.sum(rows) / rows.size))
    rec_center_col = int(round(np.sum(cols) / cols.size))
    if cols[2] == cols[1]:
        rec_ang = 90
    else:
        rec_ang = np.rad2deg(np.arctan((rows[2] - rows[1]) / (cols[2] - cols[1])))
    gripper_hh = int(round(np.sqrt((rows[1] - rows[0]) ** 2 + (cols[1] - cols[0]) ** 2) / 2))
    gripper_hw = int(round(np.sqrt((rows[3] - rows[0]) ** 2 + (cols[3] - cols[0]) ** 2) / 2))
    predicted_pose = [rec_center_row, rec_center_col, rec_ang, gripper_hh, gripper_hw]
    return predicted_pose


def evaluate_grasp(grasp, mask):
    contact_rl, contact_rr = extract_contact_region(grasp[0], grasp[1], grasp[2], grasp[3], grasp[4], mask)
    cli, trans, rotation, slippage, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr, grasp[2],
                                                                                  grasp[3], grasp[4])
    score = grasp_quality_score_v2(cli, trans, rotation, slippage, offs, grasp[3])
    return score


def show_grasps(grasp, img):
    grasp = np.array(grasp).reshape((-1, 5))
    for gi in range(grasp.shape[0]):
        ghh = grasp[gi][3]
        ghw = grasp[gi][4]
        corner_vectors = np.array([[-ghw, -ghh], [ghw, -ghh], [ghw, ghh], [-ghw, ghh]])
        new_corner_vectors = find_new_corner_vectors(corner_vectors, grasp[gi][2])
        center = [int(round(grasp[gi][1])), int(round(grasp[gi][0]))]
        new_corners = new_corner_vectors + np.array([center, center, center, center])
        for k in range(4):
            line_width = 2
            line_color = (255, 0, 0)
            if k != 3:
                if k == 0 or k == 2:
                    line_width = 1
                    line_color = (255, 255, 0)
                cv2.line(img, (new_corners[k][0], new_corners[k][1]),
                         (new_corners[k + 1][0], new_corners[k + 1][1]), line_color, line_width)
            else:
                cv2.line(img, (new_corners[3][0], new_corners[3][1]),
                         (new_corners[0][0], new_corners[0][1]), line_color, line_width)
        cv2.circle(img, (center[0], center[1]), 2, (0, 255, 0), -1)
    cv2.imshow('grasps', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


path = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'
path_c = r'D:\PhD\NNTraining\data\rawDataSet'
n = 370
data = np.loadtxt(os.path.join(path, 'IdMapping.txt'), dtype=np.int16, delimiter=',')
# print(data)
for i in tqdm(range(n)):
    inst_n = data[i, 1]
    if inst_n >= 1000:
        mask_image_name = 'pcd' + str(inst_n) + 'mask.png'
        detected_pose_file = 'dr' + str(inst_n) + '.txt'
        refined_pose_file = 'dr' + str(inst_n) + 'rf.txt'
        pos_pose_file = 'pcd' + str(inst_n) + 'cpos.txt'
        neg_pose_file = 'pcd' + str(inst_n) + 'cneg.txt'
        save_id = str(inst_n)
    else:
        mask_image_name = 'pcd0' + str(inst_n) + 'mask.png'
        detected_pose_file = 'dr0' + str(inst_n) + '.txt'
        refined_pose_file = 'dr0' + str(inst_n) + 'rf.txt'
        pos_pose_file = 'pcd0' + str(inst_n) + 'cpos.txt'
        neg_pose_file = 'pcd0' + str(inst_n) + 'cneg.txt'
        save_id = '0' + str(inst_n)

    pos_poses = np.loadtxt(os.path.join(path_c, pos_pose_file), dtype=np.float16)
    neg_poses = np.loadtxt(os.path.join(path_c, neg_pose_file), dtype=np.float16)
    detected_pose = np.loadtxt(os.path.join(path, 'detection_results', detected_pose_file), dtype=np.float16)
    refined_pose = np.loadtxt(os.path.join(path, 'refined_results2', refined_pose_file), dtype=np.float16)
    mask_img = cv2.imread(os.path.join(path, 'masks', mask_image_name))
    Im = mask_img[:, :, 0] / 255
    # print(mask_img.shape)
    # cv2.imshow('mask', mask_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    detected_pose_ca = four_pts2center_ang(detected_pose)
    # show_grasps(detected_pose_ca, mask_img)
    detected_score = evaluate_grasp(detected_pose_ca, Im)
    refined_pose_ca = four_pts2center_ang(refined_pose)
    # show_grasps(refined_pose_ca, mask_img)
    refined_score = evaluate_grasp(refined_pose_ca, Im)
    results = np.empty((0,), dtype=np.float16)
    for ip in range(int(pos_poses.shape[0] / 4)):
        pos_pose_i = pos_poses[4 * ip:4 * (ip + 1)]
        pos_pose_ca = four_pts2center_ang(pos_pose_i, True, 1)
        pos_pose_score = evaluate_grasp(pos_pose_ca, Im)
        results = np.append(results, pos_pose_score)
    results = np.append(results, -100)
    for nn in range(int(neg_poses.shape[0] / 4)):
        neg_pose_i = neg_poses[4 * nn:4 * (nn + 1)]
        neg_pose_ca = four_pts2center_ang(neg_pose_i, True, 1)
        neg_pose_score = evaluate_grasp(neg_pose_ca, Im)
        results = np.append(results, neg_pose_score)
    results = np.append(results, -100)
    results = np.append(results, detected_score)
    results = np.append(results, refined_score)
    file_name = 'dr' + save_id + 'score.txt'
    np.savetxt(os.path.join(path, 'refined_results2', file_name), results, fmt='%.4f')
