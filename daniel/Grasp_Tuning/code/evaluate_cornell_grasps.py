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


def draw_grasp_rectangles(img, poses):
    for i in range(int(poses.shape[0] / 4)):
        pose_i = four_pts2center_ang(poses[4 * i: 4 * (i + 1), :], True, 1)
        ghh = pose_i[3]
        ghw = pose_i[4]
        corner_vectors = np.array([[-ghw, -ghh], [ghw, -ghh], [ghw, ghh], [-ghw, ghh]])
        new_corner_vectors = find_new_corner_vectors(corner_vectors, pose_i[2])
        center = [int(round(pose_i[1])), int(round(pose_i[0]))]
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
    return img


def data_filter():
    path = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'
    path_c = r'D:\PhD\NNTraining\data\rawDataSet'
    n = 370
    data = np.loadtxt(os.path.join(path, 'IdMapping.txt'), dtype=np.int16, delimiter=',')
    good_ones = np.empty((0, 1), dtype=np.uint8)
    good_counter = 0
    # print(data)
    for i in range(n):
        print(i)
        inst_n = data[i, 1]
        if inst_n >= 1000:
            mask_img_name = 'pcd' + str(inst_n) + 'mask.png'
            orginal_img_name = 'pcd' + str(inst_n) + 'r.png'
            pos_pose_file = 'pcd' + str(inst_n) + 'cpos.txt'
            save_id = str(inst_n)
        else:
            mask_img_name = 'pcd0' + str(inst_n) + 'mask.png'
            orginal_img_name = 'pcd0' + str(inst_n) + 'r.png'
            pos_pose_file = 'pcd0' + str(inst_n) + 'cpos.txt'
            save_id = '0' + str(inst_n)
        img = cv2.imread(os.path.join(path_c, orginal_img_name))
        mask = cv2.imread(os.path.join(path, 'masks', mask_img_name))
        mask[:, :, 0] = mask[:, :, 0] * 0
        mask[:, :, 2] = mask[:, :, 2] * 0
        result_img = cv2.addWeighted(img, 1, mask, 0.2, 0)
        pos_poses = np.loadtxt(os.path.join(path_c, pos_pose_file), dtype=np.float16)
        show_img = draw_grasp_rectangles(result_img, pos_poses)
        cv2.imshow('green mask', show_img)
        usr_in = cv2.waitKey()
        if usr_in == ord('y'):
            good_counter += 1
            print('valid data...', good_counter)
            good_ones = np.append(good_ones, [[inst_n]], axis=0)
        cv2.destroyAllWindows()
    np.savetxt(os.path.join(path, 'good_examples.txt'), good_ones, fmt='%u')


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


def main():
    """
    evaluate average score of positive and negative grasps, evaluate grasp detection successful rate, and the rate after
    grasp refinement
    :return: results.txt
    """
    path = r'C:\Users\75077\OneDrive\2019\grasp_evaluation_ws\dataset'
    n = 370
    data = np.loadtxt(os.path.join(path, 'IdMapping.txt'), dtype=np.int16, delimiter=',')
    bad_masks = np.empty((0,))
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
        if np.amin(good_scores) < 0.05:
            bad_masks = np.append(bad_masks, inst_n)
        else:
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

    bad_mask_file = os.path.join(path, 'bad_masks.txt')  # id of the bad masks
    pn_result_file = os.path.join(path, 'pn_result.txt')
    dr_result_file = os.path.join(path, 'dr_result.txt')

    np.savetxt(bad_mask_file, bad_masks, fmt='%u')
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

    plt.figure(1)
    plt.subplot(221)
    plt.bar(np.arange(1, 12), score_hist[0, :])
    plt.xticks(np.arange(1, 12))
    plt.subplot(223)
    plt.bar(np.arange(1, 12), score_hist[1, :])
    plt.xticks(np.arange(1, 12))

    plt.subplot(222)
    plt.bar(np.arange(1, 12), valid_score_hist[0, :])
    plt.xticks(np.arange(1, 12))
    plt.subplot(224)
    plt.bar(np.arange(1, 12), valid_score_hist[1, :])
    plt.xticks(np.arange(1, 12))
    plt.show()

    plt.figure(2)
    plt.subplot(131)
    plt.title('NN detection results')
    plt.bar(np.arange(7, 10), dr_result[1:, 0] / valid_mask_count)
    plt.xticks(np.arange(7, 10))
    plt.yticks(np.round(dr_result[1:, 0] / valid_mask_count, 2))
    plt.xlabel('grasp score category')
    plt.ylabel('Percentages')
    plt.subplot(133)
    plt.title('Refined results')
    plt.bar(np.arange(7, 10), dr_result[1:, 1] / valid_mask_count)
    plt.xticks(np.arange(7, 10))
    plt.yticks(np.round(dr_result[1:, 1] / valid_mask_count, 2))
    plt.xlabel('grasp score category')
    plt.ylabel('Percentages')
    plt.show()

    print('my algorithm evaluated successful rate of NN detection: ',
          np.round(np.sum(dr_result[1:, 0]) / dr_result[0, 0] * 100, 2), '%')
    print('my algorithm evaluated successful rate after refinements: ',
          np.round(np.sum(dr_result[1:, 1]) / dr_result[0, 0] * 100, 2), '%')
    print('human evaluated successful rate of NN detection: ')
    print('human evaluated successful rate after refinements: ')


if __name__ == '__main__':
    # data_filter()
    # evaluate_all_grasps()
    main()
