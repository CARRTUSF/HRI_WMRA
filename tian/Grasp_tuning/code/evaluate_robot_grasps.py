import numpy as np
import cv2


def draw_key_points1(event, x, y, param, flag):
    global bg_keyPoints, bg_imshow
    if event == cv2.EVENT_LBUTTONDOWN:
        bg_keyPoints = np.append(bg_keyPoints, [[x, y]], axis=0)
        cv2.circle(bg_imshow, (x, y), 2, (0, 0, 255), -1)


def draw_key_points2(event, x, y, param, flag):
    global ag_keyPoints, ag_imshow
    if event == cv2.EVENT_LBUTTONDOWN:
        ag_keyPoints = np.append(ag_keyPoints, [[x, y]], axis=0)
        cv2.circle(ag_imshow, (x, y), 2, (0, 0, 255), -1)


def key_points2rot_ang(key_pts):
    pass


def key_points2trans_dist(key_pts):
    sum_pts = [0, 0]
    for pt in key_pts:
        sum_pts += pt
    pass


bg_img = np.zeros((300, 200, 3), dtype=np.uint8)
ag_img = np.zeros((200, 200, 3), dtype=np.uint8)
bg_imshow = bg_img.copy()
ag_imshow = ag_img.copy()
bg_keyPoints = np.ndarray((0, 2), dtype=np.int)
ag_keyPoints = np.ndarray((0, 2), dtype=np.int)
bg_window = 'before grasp'
ag_window = 'after grasp'
cv2.namedWindow(bg_window)
cv2.namedWindow(ag_window)
cv2.setMouseCallback(bg_window, draw_key_points1)
cv2.setMouseCallback(ag_window, draw_key_points2)
while True:
    cv2.imshow(bg_window, bg_imshow)
    cv2.imshow(ag_window, ag_imshow)
    usr_input = cv2.waitKey(20)
    if usr_input == ord('q'):
        break
    elif usr_input == ord('c'):
        bg_keyPoints = np.ndarray((0, 2), dtype=np.int)
        ag_keyPoints = np.ndarray((0, 2), dtype=np.int)
        bg_imshow = bg_img.copy()
        ag_imshow = ag_img.copy()
print(bg_keyPoints)
print(ag_keyPoints)
# calculate translation distance
if bg_keyPoints.shape[0] != 0:
    bg_center = np.array([np.sum(bg_keyPoints[:, 0]), np.sum(bg_keyPoints[:, 1])]) / bg_keyPoints.shape[0]
else:
    bg_center = np.zeros((1, 2))
if ag_keyPoints.shape[0] != 0:
    ag_center = np.array([np.sum(ag_keyPoints[:, 0]), np.sum(ag_keyPoints[:, 1])]) / ag_keyPoints.shape[0]
else:
    ag_center = np.zeros((1, 2))
distance = np.sum(np.square(ag_center - bg_center))
print("distance", distance)
# calculate rotation angle

cv2.destroyAllWindows()
