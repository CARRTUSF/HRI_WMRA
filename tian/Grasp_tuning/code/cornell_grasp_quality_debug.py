import os
import cv2
import numpy as np
from visulize_trajectory import plot_grasp_path
from helpers import extract_contact_region, cv_plot_contact_region, high_level_grasp_feature, plot_contact_profile, \
    grasp_quality_score_v2
import time


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, angle, position_change, angle_change, img, img2, Im
    if event == cv2.EVENT_LBUTTONDOWN:
        position_change = True
        x1 = x
        y1 = y
    if event == cv2.EVENT_LBUTTONUP:
        position_change = False

    if event == cv2.EVENT_RBUTTONDOWN:
        angle_change = True
        if x1 == -1:
            x1 = x
            y1 = y
    if event == cv2.EVENT_RBUTTONUP:
        angle_change = False

    if event == cv2.EVENT_MOUSEMOVE:
        if position_change:
            x1, y1 = x, y
        if angle_change:
            angle = np.rad2deg(np.arctan2(y - y1, x - x1))
        if position_change or angle_change:
            img = img2.copy()
            start_time = time.time()
            contact_rl, contact_rr = extract_contact_region(y1, x1, angle, gripper_height_h, gripper_width_h,
                                                            Im[:, :, 0])
            cli, trans, rotation, slippage, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr, angle,
                                                                                          gripper_height_h,
                                                                                          gripper_width_h)
            slippage_ang = (abs(slippage[0]) + abs(slippage[1])) / 2

            scores = grasp_quality_score_v2(cli, trans, rotation, slippage, offs, gripper_height_h)

            print(f'grasp evaluation takes {time.time() - start_time} seconds')
            cv_plot_contact_region(contact_rl, img, True, lcids)
            cv_plot_contact_region(contact_rr, img, True, rcids)
            plot_grasp_path([y1, x1], angle, gripper_height_h, gripper_width_h, img)
            plot_contact_profile(contact_rl[:, 3], 400, 180, img, lcids)
            plot_contact_profile(contact_rr[:, 3], 400, 360, img, rcids)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # prediction_text2 = f'{trans:.2f}. Translation resistance score: {scores[1][0]:.2}+-{scores[1][1]:.3f}'
            # prediction_text3 = f'{rotation:.2f}. Rotation resistance score: {scores[2][0]:.2}+-{scores[2][1]:.3f}'
            # prediction_text4 = f'{slippage_ang:.2f}, [{slippage[0]:.2f}, {slippage[1]:.2f}]. Slippage resistance score: {scores[3][0]:.2}+-{scores[3][1]:.3f}'
            # prediction_text5 = f'{offs}. Gripper force distribution score: {scores[4][0]:.2}+-{scores[4][1]:.3f}'

            score_text = f'overall scores {scores * 100:.2f} %'
            # cv2.putText(img, text, (50, 250), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(img, text2, (50, 150), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, score_text, (50, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # cv2.putText(img, prediction_text1, (50, 560), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(img, prediction_text2, (50, 70), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(img, prediction_text3, (50, 90), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(img, prediction_text4, (50, 110), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(img, prediction_text5, (50, 130), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == '__main__':
    path = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'
    inst_n = 101
    if inst_n >= 1000:
        mask_image_name = 'pcd' + str(inst_n) + 'mask.png'
        pose_file_name = 'dr' + str(inst_n) + '.txt'
        path_image_name = 'dr' + str(inst_n) + '.png'
    else:
        mask_image_name = 'pcd0' + str(inst_n) + 'mask.png'
        pose_file_name = 'dr0' + str(inst_n) + '.txt'
        path_image_name = 'dr0' + str(inst_n) + '.png'
    # print(os.path.join(path, mask_image_name))
    img = cv2.imread(os.path.join(path, 'masks', mask_image_name))
    predicted_pose = np.loadtxt(os.path.join(path, 'detection_results', pose_file_name), dtype=np.int16)
    # predicted_pose = [[row col], ...]
    rows = predicted_pose[:, 0]
    cols = predicted_pose[:, 1]
    rec_center_row = int(round(np.sum(rows) / rows.size))
    rec_center_col = int(round(np.sum(cols) / cols.size))
    rec_ang = np.rad2deg(np.arctan((rows[2] - rows[1]) / (cols[2] - cols[1])))
    predicted_pose = [rec_center_row, rec_center_col, rec_ang, -1]
    gripper_height_h = int(round(np.sqrt((rows[1] - rows[0]) ** 2 + (cols[1] - cols[0]) ** 2) / 2))
    gripper_width_h = int(round(np.sqrt((rows[3] - rows[0]) ** 2 + (cols[3] - cols[0]) ** 2) / 2))

    # img_rgb = 'dr0' + str(inst_n) + '.png'
    # img_mask = 'pcd0'+str(inst_n) + 'mask.png'
    # I = cv2.imread(os.path.join(path, 'detection_results', img_rgb))
    # img = cv2.imread(os.path.join(path, 'masks', img_mask))

    windowName = 'Draw grasp rectangle'
    Im = img / 255
    img2 = img.copy()
    position_change = False
    angle_change = False
    x1 = -1
    y1 = -1
    angle = 0
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, draw_rectangle)
    while True:
        cv2.imshow(windowName, img)
        if cv2.waitKey(20) == ord('q'):
            break
    cv2.destroyAllWindows()
