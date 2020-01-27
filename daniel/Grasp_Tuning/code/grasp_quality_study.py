import os
import cv2
import numpy as np
from visulize_trajectory import plot_grasp_path
from helpers import extract_contact_region, cv_plot_contact_region, high_level_grasp_feature, plot_contact_profile, \
    grasp_quality_score, grasp_quality_score_v2
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
            contact_rl, contact_rr = extract_contact_region(y1, x1, angle, 19, 130, Im[:, :, 0])
            cli, trans, rotation, slippage, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr, angle,
                                                                                          19, 130)
            slippage_ang = (abs(slippage[0]) + abs(slippage[1])) / 2
            scores = grasp_quality_score(cli, trans, rotation, slippage, offs, gripper_hh=19)
            score2 = grasp_quality_score_v2(cli, trans, rotation, slippage, offs, 19)
            print(f'grasp evaluation takes {time.time() - start_time} seconds')
            cv_plot_contact_region(contact_rl, img, True, lcids)
            cv_plot_contact_region(contact_rr, img, True, rcids)
            plot_grasp_path([y1, x1], angle, 19, 130, img)
            plot_contact_profile(contact_rl[:, 3], 450, 400, img, lcids)
            plot_contact_profile(contact_rr[:, 3], 450, 800, img, rcids)
            font = cv2.FONT_HERSHEY_SIMPLEX

            prediction_text1 = f'Grasp with the current pose will:'
            prediction_text2 = f'move the object {trans:.2f} pixels in the image space. Translation resistance score: {scores[1][0]:.2}+-{scores[1][1]:.3f}'
            prediction_text3 = f'rotate the object {rotation:.2f} degrees. Rotation resistance score: {scores[2][0]:.2}+-{scores[2][1]:.3f}'
            prediction_text4 = f'have a maximum contact angle {slippage_ang:.2f}, {slippage[0]:.2f}, {slippage[1]:.2f} degrees. Slippage resistance score: {scores[3][0]:.2}+-{scores[3][1]:.3f}'
            prediction_text5 = f'have a contact region center offset {offs} pixels. Gripper force distribution score: {scores[4][0]:.2}+-{scores[4][1]:.3f}'

            score_text = f'overall scores {scores[0][0] * 100:.2f}+-{scores[0][1] * 100:.3f} %'
            score2_text = f'overall scores {score2 * 100:.2f} %'
            # cv2.putText(img, text, (50, 250), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(img, text2, (50, 150), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, score_text, (50, 100), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, score2_text, (50, 120), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, prediction_text1, (50, 560), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, prediction_text2, (50, 590), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, prediction_text3, (50, 620), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, prediction_text4, (50, 650), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, prediction_text5, (50, 680), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    img_rgb = 'spoon3.png'
    img_mask = 'spoon3_mask.png'
    I = cv2.imread(os.path.join(path, 'pictures', img_rgb))
    Im = cv2.imread(os.path.join(path, 'pictures', img_mask))

    windowName = 'Draw grasp rectangle'
    img = Im * 255
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
