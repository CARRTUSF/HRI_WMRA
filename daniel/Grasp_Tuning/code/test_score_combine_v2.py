from helpers import combine_score_v2, combine_scores_current
import matplotlib.pyplot as plt
import numpy as np

# n_samples = 100
# final_scores = np.ndarray((0, 4), dtype=np.float16)
# for j in range(10):
#     min_score = 0.1 * j
#     for i in range(n_samples):
#         other_scores = np.random.rand(1, 3)
#         other_scores_corrected = other_scores + (np.ones((1, 3)) - other_scores)*min_score
#         scores = np.concatenate(([[min_score]], other_scores_corrected), axis=1)
#         if np.amin(scores) < min_score:
#             print('wrong data************')
#             print(min_score)
#             print(scores)
#             print('************')
#             break
#         final_score = combine_score_v2(scores)
#         old_score = combine_scores_current(scores)
#         avg_score = np.average(scores)
#         print(final_scores.shape)
#         print(old_score)
#         final_scores = np.append(final_scores, [[min_score, avg_score, old_score, final_score]], axis=0)
# np.savetxt('newScoringTest.txt', final_scores, fmt='%.4f')
# plt.figure()
# plt.plot(final_scores[:, 1], final_scores[:, 1])
# plt.plot(final_scores[:, 1], final_scores[:, 2])
# plt.plot(final_scores[:, 1], final_scores[:, 3])
# plt.show()


n_samples = 500
weighted_sum_prop = np.ndarray((0, 4), dtype=np.float16)
for j in range(10):
    min_score = 0.1 * j
    final_scores = np.empty((0,))
    for i in range(n_samples):
        other_scores = np.random.rand(1, 3)
        other_scores_corrected = other_scores + (np.ones((1, 3)) - other_scores) * min_score
        scores = np.concatenate(([[min_score]], other_scores_corrected), axis=1)
        if np.amin(scores) < min_score:
            print('wrong data************')
            print(min_score)
            print(scores)
            print('************')
            break
        old_score = combine_scores_current(scores)
        final_scores = np.append(final_scores, [old_score], axis=0)
    print(min_score)
    print(final_scores.shape)
    avg_score = np.average(final_scores)
    d_max = np.amax(final_scores) - avg_score
    d_min = avg_score - np.amin(final_scores)
    print(avg_score)
    print(d_max)
    print(d_min)
    weighted_sum_prop = np.append(weighted_sum_prop, [[min_score, avg_score, d_max, d_min]], axis=0)

np.savetxt('WS_scoring_test.txt', weighted_sum_prop, fmt='%.4f')
plt.figure()
plt.scatter(weighted_sum_prop[:, 0], weighted_sum_prop[:, 1])
plt.show()
