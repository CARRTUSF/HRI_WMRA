import cv2
import numpy as np
import os
from visulize_trajectory import save_path_image
from vgt3 import refine_gripper_pose
from tqdm import tqdm


def refinement_cornell_dataset():
    path = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'
    n = 370
    data = np.loadtxt(os.path.join(path, 'IdMapping.txt'), dtype=np.int16, delimiter=',')
    # print(data)
    for i in tqdm(range(n)):
        inst_n = data[i, 1]
        if inst_n >= 1000:
            mask_image_name = 'pcd' + str(inst_n) + 'mask.png'
            pose_file_name = 'dr' + str(inst_n) + '.txt'
            path_image_name = 'dr' + str(inst_n) + '.png'
            center_file_name = 'pcd' + str(inst_n) + 'ct.txt'
            save_id = str(inst_n)
        else:
            mask_image_name = 'pcd0' + str(inst_n) + 'mask.png'
            pose_file_name = 'dr0' + str(inst_n) + '.txt'
            path_image_name = 'dr0' + str(inst_n) + '.png'
            center_file_name = 'pcd0' + str(inst_n) + 'ct.txt'
            save_id = '0' + str(inst_n)
        # print(os.path.join(path, mask_image_name))
        mask_img = cv2.imread(os.path.join(path, 'masks', mask_image_name))
        predicted_pose = np.loadtxt(os.path.join(path, 'detection_results', pose_file_name), dtype=np.int16)
        # predicted_pose = [[row col], ...]
        rows = predicted_pose[:, 0]
        cols = predicted_pose[:, 1]
        rec_center_row = int(round(np.sum(rows) / rows.size))
        rec_center_col = int(round(np.sum(cols) / cols.size))
        if cols[2] == cols[1]:
            rec_ang = 90
        else:
            rec_ang = np.rad2deg(np.arctan((rows[2] - rows[1]) / (cols[2] - cols[1])))
        predicted_pose = [rec_center_row, rec_center_col, rec_ang, -1]
        gripper_height_h = int(round(np.sqrt((rows[1] - rows[0]) ** 2 + (cols[1] - cols[0]) ** 2) / 2))
        gripper_width_h = int(round(np.sqrt((rows[3] - rows[0]) ** 2 + (cols[3] - cols[0]) ** 2) / 2))
        # print(predicted_pose, gripper_height_h, gripper_width_h)
        # print(mask_img.shape)
        # cv2.imshow('mask', mask_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        trajectory = refine_gripper_pose(mask_img / 255, predicted_pose, gripper_height_h, gripper_width_h)
        # print(trajectory)
        if trajectory[-1, 3] == -1:  # no object detected
            mask_center = np.loadtxt(os.path.join(path, 'masks', center_file_name))
            predicted_pose[0] = mask_center[0]
            predicted_pose[1] = mask_center[1]
            trajectory = refine_gripper_pose(mask_img / 255, predicted_pose, gripper_height_h, gripper_width_h)
        path_image = cv2.imread(os.path.join(path, 'detection_results', path_image_name))
        scores_file = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset\refined_results\dr' + save_id + 'sc.txt'
        scores = [trajectory[0, 3], trajectory[-1, 3]]
        np.savetxt(scores_file, scores, fmt='%.4f')
        trajectory = np.array(trajectory[-1, :]).reshape((1, -1))
        # print(trajectory)
        save2file = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset\refined_results\dr' + save_id + 'rf'
        save_path_image(trajectory, path_image, save2file, gripper_height_h, gripper_width_h, save_rect=True,
                        init_score=scores[0])


if __name__ == '__main__':
    refinement_cornell_dataset()
