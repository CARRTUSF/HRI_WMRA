import cv2
import numpy as np
import os
from visulize_trajectory import save_path_image
from helpers import extract_contact_region, high_level_grasp_feature, grasp_quality_score_v2
from visulize_trajectory import find_new_corner_vectors
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def evaluate_all_grasps():
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
        refined_pose = np.loadtxt(os.path.join(path, 'refined_results', refined_pose_file), dtype=np.float16)
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
        results = np.append(results, -100)
        results = np.append(results, refined_score)
        file_name = 'dr' + save_id + 'scores.txt'
        np.savetxt(os.path.join(path, 'refined_results', file_name), results, fmt='%.4f')


def draw_grasp_rectangles(img, poses, color1=(255, 0, 0), color2=(255, 255, 0), rc_switch=False, order_shift=0):
    for i in range(int(poses.shape[0] / 4)):
        pose_i = four_pts2center_ang(poses[4 * i: 4 * (i + 1), :], rc_switch, order_shift)
        ghh = pose_i[3]
        ghw = pose_i[4]
        corner_vectors = np.array([[-ghw, -ghh], [ghw, -ghh], [ghw, ghh], [-ghw, ghh]])
        new_corner_vectors = find_new_corner_vectors(corner_vectors, pose_i[2])
        center = [int(round(pose_i[1])), int(round(pose_i[0]))]
        new_corners = new_corner_vectors + np.array([center, center, center, center])
        for k in range(4):
            line_width = 2
            line_color = color1
            if k != 3:
                if k == 0 or k == 2:
                    line_width = 1
                    line_color = color2
                cv2.line(img, (new_corners[k][0], new_corners[k][1]),
                         (new_corners[k + 1][0], new_corners[k + 1][1]), line_color, line_width)
            else:
                cv2.line(img, (new_corners[3][0], new_corners[3][1]),
                         (new_corners[0][0], new_corners[0][1]), line_color, line_width)
            cv2.circle(img, (center[0], center[1]), 2, (0, 255, 0), -1)
    return img


def human_evaluate(eval_type, save_file):
    path = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'
    # path_c = r'D:\PhD\NNTraining\data\rawDataSet'
    path_c = r'E:\NNTraining\data\rawDataSet'
    data = np.loadtxt(os.path.join(path, 'good_masks.txt'), dtype=np.int16)
    successful_ones = np.empty((0,), dtype=np.uint16)
    low_successful_ones = np.empty((0,), dtype=np.uint16)
    success_counter = 0
    low_success_counter = 0
    print(data)
    for i in range(data.shape[0]):
        print(i)
        inst_n = data[i]
        print(inst_n)
        if inst_n >= 1000:
            mask_img_name = 'pcd' + str(inst_n) + 'mask.png'
            orginal_img_name = 'pcd' + str(inst_n) + 'r.png'
            pos_pose_file = 'pcd' + str(inst_n) + 'cpos.txt'
            detected_pose_file = 'dr' + str(inst_n) + '.txt'
            refined_pose_file = 'dr' + str(inst_n) + 'rf.txt'
        else:
            mask_img_name = 'pcd0' + str(inst_n) + 'mask.png'
            orginal_img_name = 'pcd0' + str(inst_n) + 'r.png'
            pos_pose_file = 'pcd0' + str(inst_n) + 'cpos.txt'
            detected_pose_file = 'dr0' + str(inst_n) + '.txt'
            refined_pose_file = 'dr0' + str(inst_n) + 'rf.txt'

        img = cv2.imread(os.path.join(path_c, orginal_img_name))
        mask = cv2.imread(os.path.join(path, 'masks', mask_img_name))
        mask[:, :, 0] = mask[:, :, 0] * 0
        mask[:, :, 2] = mask[:, :, 2] * 0
        result_img = cv2.addWeighted(img, 1, mask, 0.2, 0)
        pos_poses = np.loadtxt(os.path.join(path_c, pos_pose_file), dtype=np.float16)
        pred_pose = np.loadtxt(os.path.join(path, 'detection_results', detected_pose_file), dtype=np.float16)
        refined_pose = np.loadtxt(os.path.join(path, 'refined_results', refined_pose_file), dtype=np.float16)

        if eval_type == 1:
            print('evaluate predicted grasp...')
            show_img = draw_grasp_rectangles(result_img, pred_pose, (0, 0, 255), (200, 0, 200))
            img_cp = np.copy(show_img)
            show_img2 = draw_grasp_rectangles(img_cp, pos_poses, rc_switch=True, order_shift=1)
        elif eval_type == 2:
            print('evaluate refined grasp...')
            show_img = draw_grasp_rectangles(result_img, refined_pose, (0, 100, 255), (200, 100, 100))
            img_cp = np.copy(show_img)
            show_img2 = draw_grasp_rectangles(img_cp, pos_poses, rc_switch=True, order_shift=1)
        else:
            show_img = draw_grasp_rectangles(result_img, np.empty((0,)))
            show_img2 = draw_grasp_rectangles(result_img, pos_poses)
        # show_img = draw_grasp_rectangles(result_img, pos_poses)
        cv2.imshow('show grasps', show_img)
        while True:
            usr_in = cv2.waitKey()
            if usr_in == ord('y'):
                success_counter += 1
                print('success...', success_counter)
                successful_ones = np.append(successful_ones, inst_n)
                break
            if usr_in == ord('u'):
                low_success_counter += 1
                print('success...', low_success_counter)
                low_successful_ones = np.append(low_successful_ones, inst_n)
                break
            elif usr_in == ord('n'):
                print('fail...')
                break
            elif usr_in == ord('i'):
                print('show positives')
                cv2.imshow('show grasps', show_img2)
        cv2.destroyAllWindows()
    np.savetxt(os.path.join(path, save_file), successful_ones, fmt='%u')


def categorize_score(categories, score):
    score_char = list(str(score))
    if score_char[0] == '-':
        categories[10] += 1
    else:
        category = score_char[2]
        categories[int(category)] += 1
    return categories


def extract_score_distribution(distribution, scores1, scores2):
    for score1 in scores1:
        distribution[0, :] = categorize_score(distribution[0, :], score1)
    for score2 in scores2:
        distribution[1, :] = categorize_score(distribution[1, :], score2)
    return distribution


def draw_table(data_array, title_array=None, cell_width=5):
    table_frame_str = ''
    table_str = np.empty((), dtype=np.str)
    for i in range(data_array.shape[1] * (cell_width + 3)):
        table_frame_str += '-'
    for j in range(data_array.shape[0]):
        data_str_j = ''
        for item in data_array[j, :]:
            item = np.round(item, 1)
            item_chars = list(str(item))
            if len(item_chars) >= cell_width - 1:
                item_str = ''.join(item_chars[:cell_width - 1])
            else:
                item_str = ''.join(item_chars)
            spaces_n = cell_width - len(item_chars)
            if spaces_n > 0:
                for k in range(spaces_n):
                    item_str += " "
            data_str_j += item_str + " " + '|' + " "
        data_str_j = " " + data_str_j + '\n'
        table_str = np.append(table_str, data_str_j)
    table_title_str = ''
    if title_array is not None:
        for title in title_array:
            title_chars = list(title)
            if len(title_chars) >= cell_width - 1:
                title_str = ''.join(title_chars[:cell_width - 1])
            else:
                title_str = ''.join(title_chars)
            spaces_n = cell_width - len(title_chars)
            if spaces_n > 0:
                for kk in range(spaces_n):
                    title_str += " "
            table_title_str += title_str + " " + '|' + " "
    print_str = " " + table_title_str + '\n' + '+' + table_frame_str + '+' + '\n'
    for n in range(table_str.shape[0]):
        print_str += table_str[n]
    print_str += '-' + table_frame_str + '-'
    return print_str


def eval_human_data():
    path = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'
    success = np.loadtxt(os.path.join(path, 'predict_success_tian.txt'), dtype=np.int16)
    all_data = np.loadtxt(os.path.join(path, 'good_masks.txt'), dtype=np.int16)
    rate = np.round(success.shape[0] / all_data.shape[0] * 100, 2)
    print('human eval:', rate, '%')
    detected_scores = np.empty((0, 1))
    refined_scores = np.empty((0, 1))
    for i in range(success.shape[0]):
        inst_n = success[i]
        if inst_n >= 1000:
            mask_img_name = 'pcd' + str(inst_n) + 'mask.png'
            detected_pose_file = 'dr' + str(inst_n) + '.txt'
            refined_pose_file = 'dr' + str(inst_n) + 'rf.txt'
        else:
            mask_img_name = 'pcd0' + str(inst_n) + 'mask.png'
            detected_pose_file = 'dr0' + str(inst_n) + '.txt'
            refined_pose_file = 'dr0' + str(inst_n) + 'rf.txt'
        detected_pose = np.loadtxt(os.path.join(path, 'detection_results', detected_pose_file), dtype=np.float16)
        refined_pose = np.loadtxt(os.path.join(path, 'refined_results', refined_pose_file), dtype=np.float16)
        mask_img = cv2.imread(os.path.join(path, 'masks', mask_img_name))
        Im = mask_img[:, :, 0] / 255
        detected_pose_ca = four_pts2center_ang(detected_pose)
        # show_grasps(detected_pose_ca, mask_img)
        detected_score = evaluate_grasp(detected_pose_ca, Im)
        refined_pose_ca = four_pts2center_ang(refined_pose)
        # show_grasps(refined_pose_ca, mask_img)
        refined_score = evaluate_grasp(refined_pose_ca, Im)
        detected_scores = np.append(detected_scores, [[detected_score]], axis=0)
        refined_scores = np.append(refined_scores, refined_score)
    print(detected_scores)
    np.savetxt(os.path.join(path, 'he_detected_scores.txt'), detected_scores, fmt='%.4f')
    # np.savetxt(os.path.join(path, 'he_refined_scores.txt'), refined_score, fmt='%.4f')


def main():
    """
    evaluate average score of positive and negative grasps, evaluate grasp detection successful rate, and the rate after
    grasp refinement
    :return: results.txt
    """
    path = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'
    n = 370
    data = np.loadtxt(os.path.join(path, 'IdMapping.txt'), dtype=np.int16, delimiter=',')
    good_masks = np.empty((0,))
    valid_mask_count = 0
    good_eval_count = 0.0
    valid_scores_sum = np.zeros((2,))
    # dr_result
    # [ detect_total     refine_total
    #   detect_success1  refine_success1    0.9
    #   detect_success2  refine_success2    0.8
    #   detect_success3  refine_success3]   0.7
    dr_result = np.zeros((4, 2))
    score_hist = np.zeros((2, 11), dtype=np.uint16)  # [0-0.1, 0.1-0.2, 0.2-0.3, ...]
    score_count = [0, 0]
    valid_score_hist = np.zeros((2, 11), dtype=np.uint16)
    valid_score_count = [0, 0]
    for i in tqdm(range(n)):
        inst_n = data[i, 1]
        if inst_n >= 1000:
            scores_file = 'dr' + str(inst_n) + 'scores.txt'
            # save_id = str(inst_n)
        else:
            scores_file = 'dr0' + str(inst_n) + 'scores.txt'
            # save_id = '0' + str(inst_n)
        # extract scores -------------------------------------------------------
        scores = np.loadtxt(os.path.join(path, 'refined_results', scores_file))
        score_pointer = 0
        good_scores = np.empty((0,))
        bad_scores = np.empty((0,))
        for g in range(scores.shape[0]):
            if scores[g] == -100:
                score_pointer = g
                break
            good_scores = np.append(good_scores, scores[g])
        for b in range(score_pointer + 1, scores.shape[0]):
            if scores[b] == -100:
                score_pointer = b
                break
            bad_scores = np.append(bad_scores, scores[b])
        predicted_score = scores[score_pointer + 1]
        refined_score = scores[score_pointer + 3]
        # ----------------------------------------------------------------------
        # calculate scores distribution *****************************************
        current_inst_score_hist = extract_score_distribution(np.zeros((2, 11), dtype=np.uint16), good_scores,
                                                             bad_scores)
        score_hist += current_inst_score_hist
        score_count[0] += good_scores.shape[0]
        score_count[1] += bad_scores.shape[0]
        if np.amin(good_scores) > 0.05:
            good_masks = np.append(good_masks, inst_n)
            valid_score_hist += current_inst_score_hist
            valid_score_count[0] += good_scores.shape[0]
            valid_score_count[1] += bad_scores.shape[0]
            # ***********************************************************************
            # for valid masks evaluate the algorithm ////////////////////////////
            valid_mask_count += 1
            valid_scores_sum[0] += np.average(good_scores)
            valid_scores_sum[1] += np.average(bad_scores)

            if np.average(good_scores) > 0.8 and np.average(bad_scores) < 0.3:
                good_eval_count += 1

            if predicted_score > 0.9:
                dr_result[3, 0] += 1
            elif predicted_score > 0.8:
                dr_result[2, 0] += 1
            elif predicted_score > 0.7:
                dr_result[1, 0] += 1

            if refined_score > 0.9:
                dr_result[3, 1] += 1
            elif refined_score > 0.8:
                dr_result[2, 1] += 1
            elif refined_score > 0.7:
                dr_result[1, 1] += 1
    dr_result[0, :] += [valid_mask_count, valid_mask_count]
    pn_result = np.append(valid_scores_sum, good_eval_count)
    pn_result = pn_result / valid_mask_count
    # pn_result: average positive score , average negative score, percentage of which positive>0.8 and negative<0.3, valid masks count
    pn_result = np.append(pn_result, valid_mask_count)

    good_mask_file = os.path.join(path, 'good_masks.txt')  # id of the bad masks
    pn_result_file = os.path.join(path, 'pn_result.txt')
    dr_result_file = os.path.join(path, 'dr_result.txt')

    np.savetxt(good_mask_file, good_masks, fmt='%u')
    np.savetxt(pn_result_file, pn_result, fmt='%.4f')
    np.savetxt(dr_result_file, dr_result, fmt='%.4f')

    print("total grasp poses (good ones, bad ones): ", score_count)
    print("total grasp poses of valid instances (good ones, bad ones): ", valid_score_count)
    score_count = np.array(score_count).reshape((2, 1))
    valid_score_count = np.array(valid_score_count).reshape((2, 1))
    score_hist_post = (score_hist / score_count) * 100
    valid_score_hist_post = (valid_score_hist / valid_score_count) * 100
    title = np.array(
        ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0',
         '-1'])
    print('^^^^  cornell grasp dataset 370 instances ground truth grasp score distribution  ^^^^')
    print(draw_table(score_hist_post, title, cell_width=8))
    print(
        '^^^^  cornell grasp dataset 225 instances(with valid mask detection) ground truth grasp score distribution  ^^^^')
    print(draw_table(valid_score_hist_post, title, cell_width=8))

    fig1, axs1 = plt.subplots(1, 2, constrained_layout=True)
    fig1.suptitle('Grasp score distribution of 370 instances from Cornell grasping dataset')
    axs1[0].bar(np.arange(0, 100, 10), score_hist[0, :-1], width=8)
    axs1[0].set_xticks(np.arange(0, 100, 10))
    axs1[0].set_xlabel('grasp success chance')
    axs1[0].set_ylabel('number of grasps')
    plt.sca(axs1[0])
    plt.sca(axs1[1])
    axs1[0].set_title('2250 ground truth positive grasps')

    axs1[1].bar(np.arange(0, 100, 10), score_hist[1, :-1], width=8)
    axs1[1].set_xticks(np.arange(0, 100, 10))
    axs1[1].set_xlabel('grasp success chance')
    axs1[1].set_ylabel('number of grasps')
    axs1[1].set_title('1199 ground truth negative grasps')

    fig2, axs2 = plt.subplots(1, 2, constrained_layout=True)
    fig2.suptitle('Grasp score distribution of 225 instances from Cornell grasping dataset with valid masks')
    plt.sca(axs2[0])
    plt.sca(axs2[1])
    axs2[0].bar(np.arange(0, 100, 10), valid_score_hist[0, :-1], width=8)
    axs2[0].set_xticks(np.arange(0, 100, 10))
    axs2[0].set_xlabel('grasp success chance')
    axs2[0].set_ylabel('number of grasps')
    axs2[0].set_title('1051 ground truth positive grasps')

    axs2[1].bar(np.arange(0, 100, 10), valid_score_hist[1, :-1], width=8)
    axs2[1].set_xticks(np.arange(0, 100, 10))
    axs2[1].set_xlabel('grasp success chance')
    axs2[1].set_ylabel('number of grasps')
    axs2[1].set_title('710 ground truth negative grasps')

    fig3, axs3 = plt.subplots(1, 2, constrained_layout=True)
    fig3.suptitle('NN detection and refinement evaluation')
    plt.sca(axs3[0])
    plt.sca(axs3[1])
    axs3[0].bar(np.arange(7, 10), dr_result[1:, 0] / valid_mask_count * 100)
    axs3[0].set_xticks(np.arange(7, 10))
    axs3[0].set_yticks(np.round(dr_result[1:, 0] / valid_mask_count * 100, 2))
    axs3[0].set_xlabel('grasp success chance')
    axs3[0].set_ylabel('percentage of grasps')
    axs3[0].set_title('Ian Lenz 2013 NN detection result')

    axs3[1].bar(np.arange(7, 10), dr_result[1:, 1] / valid_mask_count * 100)
    axs3[1].set_xticks(np.arange(7, 10))
    axs3[1].set_yticks(np.round(dr_result[1:, 1] / valid_mask_count * 100, 2))
    axs3[1].set_xlabel('grasp success chance')
    axs3[1].set_ylabel('percentage of grasps')
    axs3[1].set_title('Refined NN detection result')

    # plt.figure(2)
    # plt.subplot(131)
    # plt.title('NN detection results')
    # plt.bar(np.arange(7, 10), dr_result[1:, 0] / valid_mask_count)
    # plt.xticks(np.arange(7, 10))
    # plt.yticks(np.round(dr_result[1:, 0] / valid_mask_count, 2))
    # plt.xlabel('grasp score category')
    # plt.ylabel('Percentages')
    # plt.subplot(133)
    # plt.title('Refined results')
    # plt.bar(np.arange(7, 10), dr_result[1:, 1] / valid_mask_count)
    # plt.xticks(np.arange(7, 10))
    # plt.yticks(np.round(dr_result[1:, 1] / valid_mask_count, 2))
    # plt.xlabel('grasp score category')
    # plt.ylabel('Percentages')
    plt.show()

    print('my algorithm evaluated successful rate of NN detection: ',
          np.round(np.sum(dr_result[1:, 0]) / dr_result[0, 0] * 100, 2), '%')
    print('my algorithm evaluated successful rate after refinements: ',
          np.round(np.sum(dr_result[1:, 1]) / dr_result[0, 0] * 100, 2), '%')
    print('human evaluated successful rate of NN detection: ')
    print('human evaluated successful rate after refinements: ')
    print(good_masks.shape)


if __name__ == '__main__':
    # data_filter()
    # evaluate_all_grasps()
    main()
    # human_evaluate(1, 'predict_success_tian.txt')
    # human_evaluate(2, 'refine_success_tian.txt'))
    # eval_human_data()
