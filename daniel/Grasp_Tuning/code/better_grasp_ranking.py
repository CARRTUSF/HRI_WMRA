import numpy as np
import cv2
from helpers import plot_grasp_rect, extract_contact_region, high_level_grasp_feature, grasp_quality_score
import os


def compare(grasp1, grasp2, grasp3, img):
    img_s = np.copy(img)
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    plot_grasp_rect(grasp1, 19, 130, img_s, red)
    plot_grasp_rect(grasp2, 19, 130, img_s, green)
    plot_grasp_rect(grasp3, 19, 130, img_s, blue)
    cv2.imshow('Is green between red and blue', img_s)
    while True:
        u_in = cv2.waitKey(0)
        if u_in == ord('1') or u_in == ord('2') or u_in == ord('0') or u_in == ord('3') or u_in == ord(
                'q') or u_in == ord('e') or u_in == ord('c'):
            break
        else:
            print('press 1 move left, 2 move right, 0 confirm, 3 100%fail, q quit')
    cv2.destroyAllWindows()
    return u_in


def get_grasp_scores(grasp, mask_I):
    contact_rl, contact_rr = extract_contact_region(grasp[0], grasp[1], grasp[2], 19, 130, mask_I[:, :, 0])
    cli, trans, rotation, slippage, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr, grasp[2], 19,
                                                                                  130)
    slippage_ang = (abs(slippage[0]) + abs(slippage[1])) / 2
    print('[[[[[[[[[[[[[[[[[[[[[[[[[[')
    print('translation', trans)
    print('rotation', rotation)
    print('slippage angle', slippage_ang)
    print('force center off', offs)

    scores = grasp_quality_score(cli, trans, rotation, slippage, offs, 19)
    # w4 = max(2 * scores[2][0] - 1.1, 0.1)
    # s24 = (1 - w4) * scores[2][0] + w4 * scores[4][0]
    # u24 = np.sqrt(((1 - w4) * scores[2][1]) ** 2 + (w4 * scores[4][1]) ** 2)
    # print('combined s24 u24', s24, u24)
    return [scores[0][0], scores[1][0], scores[2][0], scores[3][0], scores[4][0]]


def show_grasps(grasp1, grasp2, grasp3, rank, img_m):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    show_image = img_m * 255
    plot_grasp_rect(grasp1, 19, 130, show_image, red)
    plot_grasp_rect(grasp2, 19, 130, show_image, green)
    plot_grasp_rect(grasp3, 19, 130, show_image, blue)
    grasp_scores = get_grasp_scores(grasp2, img_m)
    grasp_scores_pre = get_grasp_scores(grasp1, img_m)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'{rank}'
    score_text = f'score:{grasp_scores[0]:.3f}, s1:{grasp_scores[1]:.3f}, s2:{grasp_scores[2]:.3f}, s3:{grasp_scores[3]:.3f}, s4:{grasp_scores[4]:.3f}'
    score_text_pre = f'score:{grasp_scores_pre[0]:.3f}, s1:{grasp_scores_pre[1]:.3f}, s2:{grasp_scores_pre[2]:.3f}, s3:{grasp_scores_pre[3]:.3f}, s4:{grasp_scores_pre[4]:.3f}'
    cv2.putText(show_image, text, (100, 100), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(show_image, score_text_pre, (100, 150), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(show_image, score_text, (100, 200), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('show grasps', show_image)
    while True:
        usr_input = cv2.waitKey(0)
        if usr_input == ord('1') or usr_input == ord('2') or usr_input == ord('q') or usr_input == ord('g'):
            cv2.destroyAllWindows()
            break
    return usr_input


def add_new_grasp2(results_array, data_i, img):
    place = 0
    row = data_i[-3]
    col = data_i[-2]
    ang = data_i[-1]
    gi = [row, col, ang]
    quit_all = False
    while True:
        if place == 0:
            print('left end', place)
            g_left = [0, 0, 0]
            g_right = results_array[place, -3:]
        elif place == results_array.shape[0]:
            print('right end', place)
            g_left = results_array[place - 1, -3:]
            g_right = [0, 0, 0]
        else:
            print('mid', place)
            g_left = results_array[place - 1, -3:]
            g_right = results_array[place, -3:]

        usr_input = compare(g_left, gi, g_right, img)
        if usr_input == ord('1'):
            place = max(place - 1, 0)
            print('move left', place)
        elif usr_input == ord('2'):
            place = min(place + 1, results_array.shape[0])
            print('move right', place)
        elif usr_input == ord('3'):
            print('skip')
            break
        # elif usr_input == ord('e'):
        #     xi = ''
        #     while True:
        #         try:
        #             xi = int(xi)
        #             break
        #         except ValueError:
        #             xi = input(f'enter the estimated rank of the grasp, must be an integer in range[0, {results_array.shape[0]}]')
        #     xi = min(xi, results_array.shape[0])
        #     xi = max(xi, 0)
        #     place = xi
        elif usr_input == ord('c'):
            data_c = results_array[place, 1:]
            print('data_c', data_c)
            results_array[place:, 0] -= 1
            results_array = np.concatenate([results_array[:place, :], results_array[place + 1:, :]], axis=0)
            results_array, s = add_new_grasp2(results_array, data_c, img)
            print('done-----------------------------------')
        elif usr_input == ord('0'):
            print('place confirmed', place)
            new_grasp_data = np.concatenate([[place], data_i])
            results_array[place:, 0] += 1
            results_array = np.concatenate([results_array[:place, :], [new_grasp_data], results_array[place:, :]])
            # print('saved data #: ', results_array.size)
            # np.savetxt(output, results_array, fmt='%1.4f')
            break
        elif usr_input == ord('q'):
            print('quit')
            quit_all = True
            break

    return results_array, quit_all


def rank_grasp(mask_image, datafile, output_file):
    x = np.loadtxt(datafile)
    ids = np.arange(x.shape[0]).reshape(-1, 1)
    x = np.concatenate([ids, x], axis=1)
    results = np.ndarray((0, x.shape[1] + 1), dtype=np.float16)
    first_grasp_data = np.concatenate([[0], x[0]])
    results = np.append(results, [first_grasp_data], axis=0)
    for i in range(1, x.shape[0]):
        data_i = x[i]
        results, s = add_new_grasp2(results, data_i, mask_image)
        print('progress', results.shape[0])
        if s:
            break
    np.savetxt(output_file, results, fmt='%1.4f')


def show_ranked(mask_image, inputdata):
    ranked_data = np.loadtxt(inputdata)
    i = 1
    while True:
        m_img = np.copy(mask_image)
        grasp_i = ranked_data[i, -3:]
        grasp_il = ranked_data[i - 1, -3:]
        grasp_ir = ranked_data[i + 1, -3:]
        current_rank = ranked_data[i, 0]
        ui = show_grasps(grasp_il, grasp_i, grasp_ir, current_rank, m_img)
        if ui == ord('1'):
            i = max(i - 1, 1)
        elif ui == ord('2'):
            i = min(i + 1, ranked_data.shape[0] - 2)
        elif ui == ord('q'):
            break


def score_ranked(mask_I, input, output1, output2):
    ranked_data = np.loadtxt(input)
    predicted_scores = np.ndarray((0, 5), dtype=np.float16)
    predicted_score_uncertainties = np.ndarray((0, 5), dtype=np.float16)
    for i in range(ranked_data.shape[0]):
        y = ranked_data[i, 2]
        x = ranked_data[i, 3]
        angle = ranked_data[i, 4]
        contact_rl, contact_rr = extract_contact_region(y, x, angle, 19, 130, mask_I[:, :, 0])
        cli, trans, rotation, slippage, lcids, rcids, offs = high_level_grasp_feature(contact_rl, contact_rr, angle, 19,
                                                                                      130)
        scores = grasp_quality_score(cli, trans, rotation, slippage, offs, 19)
        score_array = [scores[0][0], scores[1][0], scores[2][0], scores[3][0], scores[4][0]]
        uncertainty_array = [scores[0][1], scores[1][1], scores[2][1], scores[3][1], scores[4][1]]
        predicted_scores = np.append(predicted_scores, [score_array], axis=0)
        predicted_score_uncertainties = np.append(predicted_score_uncertainties, [uncertainty_array], axis=0)
    np.savetxt(output1, predicted_scores, fmt='%1.3f')
    np.savetxt(output2, predicted_score_uncertainties, fmt='%1.3f')


def rank_grasp_v2(obj_mask, grasps, random_order=True):
    samples = np.copy(grasps)
    if random_order:
        np.random.shuffle(samples)

    results = np.ndarray((0, samples.shape[1] + 1), dtype=np.float16)
    first_grasp_data = np.concatenate([[0], samples[0]])
    results = np.append(results, [first_grasp_data], axis=0)
    for i in range(1, samples.shape[0]):
        print('progress', i)
        data_i = samples[i]
        results, s = add_new_grasp2(results, data_i, obj_mask)
        if s:
            break
    return results


def find_good_grasp(obj_mask, grasps):
    i = 0
    good_grasp = np.ndarray((0, grasps.shape[1]))
    while True:
        m_img = np.copy(obj_mask)
        grasp_i = grasps[i, -3:]
        if i == 0:
            grasp_il = [0, 0, 0]
            grasp_ir = grasps[i + 1, -3:]
        elif i == grasps.shape[0] - 1:
            grasp_il = grasps[i - 1, -3:]
            grasp_ir = [0, 0, 0]
        else:
            grasp_il = grasps[i - 1, -3:]
            grasp_ir = grasps[i + 1, -3:]
        current_rank = grasps[i, 0]
        ui = show_grasps(grasp_il, grasp_i, grasp_ir, current_rank, m_img)
        if ui == ord('1'):
            i = max(i - 1, 1)
        elif ui == ord('2'):
            i = min(i + 1, grasps.shape[0] - 1)
        elif ui == ord('g'):
            good_grasp = grasps[i]
            break
        elif ui == ord('q'):
            break
    return good_grasp.reshape((-1, 5))


def rank_grasps_all_in_one(imgs, grasp_samples, output1, output2):
    human_ranked_results = np.ndarray((0, grasp_samples.shape[1] + 1))
    good_grasps = np.ndarray((0, grasp_samples.shape[1] + 1))
    for i in range(5):
        for j in range(6):
            x = input('press enter to continue, x to quit')
            if x == 'x':
                break
            si = 60 * i + 10 * j
            ei = si + 10
            sub_samples = grasp_samples[si:ei, :]  # array of grasp samples
            object_mask = imgs[i]  # string name of the object mask image
            sub_results = rank_grasp_v2(object_mask, sub_samples)
            good_grasp = find_good_grasp(object_mask / 255, sub_results)
            human_ranked_results = np.append(human_ranked_results, sub_results, axis=0)
            print(good_grasp.shape)
            print(good_grasps.shape)
            good_grasps = np.append(good_grasps, good_grasp, axis=0)
            np.savetxt(output1, human_ranked_results, fmt='%1.4f')
            np.savetxt(output2, good_grasps, fmt='%1.4f')


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    # img_rgb = 'box.png'
    img1_mask = 'spoon3_mask.png'
    img2_mask = 'box_mask.png'
    img3_mask = 'wbt_mask.png'
    img4_mask = 'spray1_mask.png'
    img5_mask = 'tennis_mask.png'

    # I = cv2.imread(os.path.join(path, 'pictures', img_rgb))
    Im1 = cv2.imread(os.path.join(path, 'pictures', img1_mask)) * 255
    Im2 = cv2.imread(os.path.join(path, 'pictures', img2_mask)) * 255
    Im3 = cv2.imread(os.path.join(path, 'pictures', img3_mask)) * 255
    Im4 = cv2.imread(os.path.join(path, 'pictures', img4_mask)) * 255
    Im5 = cv2.imread(os.path.join(path, 'pictures', img5_mask)) * 255
    mask_imgs = [Im1, Im2, Im3, Im4, Im5]
    unranked_grasp_samples = np.loadtxt('sample_grasps.txt')
    rank_grasps_all_in_one(mask_imgs, unranked_grasp_samples, 'Urvish_ranked_grasps.txt',
                           'Urvish_selected_good_grasps.txt')
    # rank_grasp(Im5, 'td6.txt', 'rtd6.txt')
    # score_ranked(Im, 'rbd6.txt', 'ap_b6.txt', 'apu_b6.txt')

    # show_ranked(Im, 'rbd6.txt')
