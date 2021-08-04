"""
This is a ROS node that subscribes to an image topic, and publishes the selected object mask image and it's bonding box.
Method: detectron2 (https://github.com/facebookresearch/detectron2)
"""
import os
import sys
import rospy
import time
import argparse

import cv2
import image_geometry
import geometry_msgs.msg
import sensor_msgs.msg
from std_msgs.msg import String
import numpy as np

# import moveit_commander

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from cv_bridge import CvBridge
from tt_grasp_evaluation import evaluate_grasp

INPUT_ROS_IMAGE = sensor_msgs.msg.Image()
MOUSE_CLICKED_AT = (-1, -1)
IMAGE = sensor_msgs.msg.Image()
CAM_INFO = sensor_msgs.msg.CameraInfo()
HAND_RANGE = sensor_msgs.msg.Range()
OBJECT_SELECTED = False
POSITION_CHANGE = False
ANGLE_CHANGE = False
X1 = -1
Y1 = -1
X2 = -1
Y2 = -1
ANGLE = 0
SCORE = -100
HW = 130
HH = 19
MASK_SHOW = np.zeros((100, 100), dtype=np.uint8)
IMG_SHOW = np.zeros((100, 100), dtype=np.uint8)
OBJECT_MASK = np.zeros((100, 100), dtype=np.uint8)


def get_hand_image(msg):
    global IMAGE
    IMAGE = msg


def get_cam_info(msg):
    global CAM_INFO
    CAM_INFO = msg


def get_end_range(msg):
    global HAND_RANGE
    HAND_RANGE = msg.range


def crop_image2roi(roi, image):
    height = image.shape[0]
    width = image.shape[1]
    roi_h, roi_w = roi
    x1 = int((width - roi_w) / 2)
    y1 = int((height - roi_h) / 2)
    if x1 < 0 or y1 < 0:
        raise Exception('ROI is larger than the image')
    else:
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        return image[y1:y2, x1:x2]


# def get_mouse_click(event, x, y, flags, param):
#     global MOUSE_CLICKED_AT
#     if event == cv2.EVENT_LBUTTONDOWN:
#         MOUSE_CLICKED_AT = (x, y)
#         print(MOUSE_CLICKED_AT)
#         print('====================')


def get_close2poi_bb_id(poi, b_boxes):
    dists = []
    x_, y_ = poi
    for box in b_boxes:
        c_x = (box[0] + box[2]) / 2
        c_y = (box[1] + box[3]) / 2
        d_x = c_x - x_
        d_y = c_y - y_
        sq_dist = d_x ** 2 + d_y ** 2
        dists.append(sq_dist)
    dists = np.array(dists)
    if dists.size == 0:
        return None
    else:
        return np.argmin(dists)


def pad_image2shape(original_img, new_img_shape):
    h0, w0 = original_img.shape
    h, w = new_img_shape
    if h0 < h and w0 < w:
        new_img = np.zeros(new_img_shape)
        x1 = int((w - w0) / 2)
        y1 = int((h - h0) / 2)
        x2 = x1 + w0
        y2 = y1 + h0
        new_img[y1:y2, x1:x2] = original_img
        return new_img
    else:
        return original_img


def get_bb_in_original_image(bb, roi, original_image_size):
    roi_h, roi_w = roi
    h, w = original_image_size
    dx = int((w - roi_w) / 2)
    dy = int((h - roi_h) / 2)
    if dx < 0 or dy < 0:
        raise Exception('ROI is larger than the image')
    else:
        # x1 = bb[0] + dx
        # y1 = bb[1] + dy
        # x2 = bb[2] + dx
        # y2 = bb[3] + dy
        return np.array(bb, dtype=int) + np.array([dx, dy, dx, dy], dtype=int)


def get_grasp_rectangle_in_cam_3d(finger_locations, z_in_cam):
    center_offset = {2: 0.017, 3: 0.027, 4: 0.036, 5: 0.046}
    left_center_d = center_offset.get(finger_locations[0], -1)
    right_center_d = center_offset.get(finger_locations[1], -1)
    grasp_half_width = (left_center_d + right_center_d) / 2
    gripper_center_in_cam = (0.013 + (left_center_d - right_center_d) / 2, -0.03825, z_in_cam)
    gripper_top_left = (gripper_center_in_cam[0] - grasp_half_width, gripper_center_in_cam[1] - 0.0065, z_in_cam)
    gripper_bot_right = (gripper_center_in_cam[0] + grasp_half_width, gripper_center_in_cam[1] + 0.0065, z_in_cam)
    return [gripper_center_in_cam, gripper_top_left, gripper_bot_right]


def get_position(event, x, y, flags, param):
    global X2, Y2, OBJECT_SELECTED
    if event == cv2.EVENT_LBUTTONUP:
        X2, Y2 = x, y
        OBJECT_SELECTED = True


def get_corners(center, theta, ghh, ghw):
    theta = np.deg2rad(theta)
    corner_vectors_original = np.array([[-ghh, ghh, ghh, -ghh], [-ghw, -ghw, ghw, ghw], [1, 1, 1, 1]])
    transformation = np.array(
        [[np.cos(theta), np.sin(theta), center[0]], [-np.sin(theta), np.cos(theta), center[1]], [0, 0, 1]])
    new_corners = np.matmul(transformation, corner_vectors_original)
    return new_corners


def plot_grasp_path(center, theta, ghh, ghw, img):
    new_corners = get_corners(center, theta, ghh, ghw)
    for k in range(4):
        if k != 3:
            cv2.line(img, (int(round(new_corners[1][k])), int(round(new_corners[0][k]))),
                     (int(round(new_corners[1][k + 1])), int(round(new_corners[0][k + 1]))), (0, 0, 255))
        else:
            cv2.line(img, (int(round(new_corners[1][3])), int(round(new_corners[0][3]))),
                     (int(round(new_corners[1][0])), int(round(new_corners[0][0]))), (0, 0, 255))
    cv2.circle(img, (center[1], center[0]), 3, (0, 255, 0), -1)


def draw_rectangle(event, x, y, flags, param):
    global X1, Y1, ANGLE, POSITION_CHANGE, ANGLE_CHANGE, MASK_SHOW, IMG_SHOW, HW, HH, SCORE, OBJECT_MASK
    if event == cv2.EVENT_LBUTTONDOWN:
        POSITION_CHANGE = True
        X1, Y1 = x, y
    if event == cv2.EVENT_LBUTTONUP:
        POSITION_CHANGE = False
    if event == cv2.EVENT_RBUTTONDOWN:
        ANGLE_CHANGE = True
        if X1 == -1:
            X1, Y1 = x, y
    if event == cv2.EVENT_RBUTTONUP:
        ANGLE_CHANGE = False

    if event == cv2.EVENT_MOUSEMOVE:
        if POSITION_CHANGE:
            X1, Y1 = x, y
        if ANGLE_CHANGE:
            ANGLE = np.rad2deg(np.arctan2(y - Y1, x - X1))

    g1 = [Y1, X1, ANGLE, HH, HW]
    SCORE = evaluate_grasp(g1, OBJECT_MASK)
    IMG_SHOW = MASK_SHOW.copy()
    plot_grasp_path([Y1, X1], ANGLE, HH, HW, IMG_SHOW)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'grasp score: ' + str(SCORE)
    cv2.putText(IMG_SHOW, text, (50, 50), font, 1, (0, 255, 0))


def find_the_closest_object(bblist, measure_point):
    min_dist = -1
    min_id = -1
    for ii in range(bblist.shape[0]):
        cr = (bblist[ii, 1] + bblist[ii, 3]) / 2
        cc = (bblist[ii, 0] + bblist[ii, 2]) / 2
        distance_squared = (measure_point[1] - cr) ** 2 + (measure_point[0] - cc) ** 2
        if ii == 0 or distance_squared < min_dist:
            min_dist = distance_squared
            min_id = ii
    if min_dist < 3000:
        return min_id
    else:
        return None


def dict2list(baxter_msg, keys):
    msg_list = []
    for key in keys:
        items = baxter_msg[key]
        for item in items:
            msg_list.append(item)
    return msg_list


def pose2trans(pose):
    trans = np.ones((4, 4))
    trans[0, 3] = pose[0]
    trans[1, 3] = pose[1]
    trans[2, 3] = pose[2]
    x = pose[3]
    y = pose[4]
    z = pose[5]
    w = pose[6]
    trans[0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
    trans[1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
    trans[2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2

    trans[0, 1] = 2 * x * y - 2 * z * w
    trans[0, 2] = 2 * x * z + 2 * y * w
    trans[1, 0] = 2 * x * y + 2 * z * w
    trans[1, 2] = 2 * z * y - 2 * x * w
    trans[2, 0] = 2 * z * x - 2 * y * w
    trans[2, 1] = 2 * z * y + 2 * x * w
    trans[3, :3] *= 0
    return trans


def main():
    global MOUSE_CLICKED_AT, IMAGE, CAM_INFO, OBJECT_SELECTED, HH, HW, SCORE, OBJECT_MASK, MASK_SHOW, IMG_SHOW
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='manual select grasp or automatically generate grasp', choices=['m', 'a'],
                        default='a')
    parser.add_argument('--limb', help='which limb of baxter to use', choices=['left', 'right'],
                        default='right')
    parser.add_argument('object_name', help='name of the object been tested')
    parser.add_argument('score_high', help='upper limit of the desired grasp score range', type=float)
    parser.add_argument('distance', help='distance between initial position and the grasp position', type=float)
    parser.add_argument('finger_locations', help='baxter gripper finger locations, location number on gripper base',
                        type=int, nargs=2)
    args = parser.parse_args()
    limb = args.limb
    hand_cam_image_tp = 'cameras/' + limb + '_hand_camera/image'
    hand_cam_info_tp = 'cameras/' + limb + '_hand_camera/camera_info'
    hand_range_tp = 'robot/range/' + limb + '_hand_range/state'
    robot_arm = limb + '_arm'
    print('starting grasp evaluation node...')
    # initialize robot----------------------------------------------------------------------------------------------
    rospy.init_node('grasp_evaluation')
    rospy.Subscriber(hand_cam_image_tp, sensor_msgs.msg.Image, callback=get_hand_image)
    rospy.Subscriber(hand_cam_info_tp, sensor_msgs.msg.CameraInfo, callback=get_cam_info)
    rospy.Subscriber(hand_range_tp, sensor_msgs.msg.Range, callback=get_end_range)

    rob_min_z = -0.193

    br = CvBridge()
    cam = image_geometry.PinholeCameraModel()
    cam_param_set = False
    usr_confirmed = False
    cam_fx = 402.437231
    cam_fy = 403.1572757
    cam_gripper_trans = np.array([[0, 1, 0, 0.03825],
                                  [-1, 0, 0, 0.012],
                                  [0, 0, 1, -0.1053],
                                  [0, 0, 0, 1]])
    # user interface**************************************
    scene_window = 'click on the object to grasp'
    cv2.namedWindow(scene_window)
    cv2.setMouseCallback(scene_window, get_position)

    mask_window = 'Selected object mask'
    cv2.namedWindow(mask_window)
    grasp_rect = None
    # trial and object specific parameters+++++++++++++++++++++++++++++++++
    desired_grasp_score = args.score_high

    # calculate projection of gripper when grasp object
    # distance subtract 0.01 to make the grasp rectangle at 1cm higher than the very tip of the gripper
    gripper_in_cam_3d = get_grasp_rectangle_in_cam_3d(args.finger_locations, args.distance - 0.01)
    # h = 0.41  # downward movement distance in meters
    cam_mf = np.array([[args.distance / cam_fx, 0, 0], [0, args.distance / cam_fy, 0], [0, 0, args.distance]])
    im_crop = [100, 700, 320, 1120]  # [row1:row2, col3:col4]
    # save file name variables +++++++++++++++++++++++++++++++++++
    save_count = 0
    file_prefix = os.getcwd() + "/data/" + args.object_name + "/" + \
                  args.object_name + "_" + str(int(desired_grasp_score * 10)) + "_"
    data = np.ndarray((0, 9), dtype=np.float16)
    object_mask_save = np.zeros((100, 100))

    # detectron2 predictor set up
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    predictor = DefaultPredictor(cfg)

    while not rospy.is_shutdown():
        # generate grasp rectangle~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if IMAGE.header.frame_id != '':
            cv_image = br.imgmsg_to_cv2(IMAGE, 'bgr8')
            if not cam_param_set:
                cam.fromCameraInfo(CAM_INFO)
                up_left = cam.project3dToPixel(gripper_in_cam_3d[1])
                bot_right = cam.project3dToPixel(gripper_in_cam_3d[2])
                HW = int(round((bot_right[0] - up_left[0]) / 2))
                HH = int(round((bot_right[1] - up_left[1]) / 2))
                cam_param_set = True
            # detect objects and their masks^^^^^^^^^^^^^^^^^^^^^^^^
            image4detection = cv_image[im_crop[0]:im_crop[1], im_crop[2]:im_crop[3], :]
            outputs = predictor(image4detection)
            predictions = outputs['instances'].to('cpu')
            v = Visualizer(image4detection[:, :, ::-1])
            out = v.draw_instance_predictions(predictions)
            boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
            masks = np.asarray(predictions.pred_masks, dtype=np.uint8) if predictions.has("pred_masks") else None
            cv2.imshow(scene_window, out.get_image()[:, :, ::-1])
            # user interface #######################################
            if boxes is not None and OBJECT_SELECTED:
                object_id = find_the_closest_object(boxes, [Y2, X2])
                object_roi = boxes[object_id]
                if object_id is None:
                    print('NO Object Detected!!')
                    OBJECT_SELECTED = False
                else:
                    OBJECT_MASK = masks[object_id]
                    # generate grasp with score in the desired range
                    MASK_SHOW = cv2.cvtColor(OBJECT_MASK * 255, cv2.COLOR_GRAY2BGR)
                    MASK_SHOW[:, :, 0] *= 0
                    MASK_SHOW[:, :, 2] *= 0
                    MASK_SHOW = cv2.addWeighted(image4detection, 1, MASK_SHOW, 0.5, 0)
                    # manual grasp selection =============================================
                    if args.mode == 'm':
                        IMG_SHOW = MASK_SHOW
                        draw_window = 'draw grasp rectangle'
                        cv2.namedWindow(draw_window)
                        cv2.setMouseCallback(draw_window, draw_rectangle)
                        while True:
                            cv2.imshow(draw_window, IMG_SHOW)
                            usr_in = cv2.waitKey(20)
                            if usr_in == 1048677:
                                usr_confirmed = True
                                OBJECT_SELECTED = False
                                grasp_rect = [Y1 + im_crop[0], X1 + im_crop[2], ANGLE]
                                cv2.destroyWindow(draw_window)
                                break
                            elif usr_in == 1048689:
                                OBJECT_SELECTED = False
                                cv2.destroyWindow(draw_window)
                                break
                    # random grasp selection ===============================================
                    else:
                        grasp_found = False
                        s_time = time.time()
                        grasp_tested = 0
                        a = 0
                        while not grasp_found:
                            gx = np.random.randint(object_roi[0], object_roi[2])
                            gy = np.random.randint(object_roi[1], object_roi[3])
                            searches_show = np.copy(MASK_SHOW)
                            cv2.circle(searches_show, (gx, gy), 2, (0, 0, 255), -1)
                            # cv2.circle(searches_show, (object_com[1], object_com[0]), 3, (255, 0, 0), -1)
                            cv2.imshow(mask_window, searches_show)
                            cv2.waitKey(1)
                            while a < 10:
                                a += 1
                                ang = np.random.randint(-90, 90)
                                grasp = [gy, gx, ang, HH, HW]
                                # print grasp
                                score, features = evaluate_grasp(grasp, OBJECT_MASK)
                                print(score)
                                print(features)
                                if 0 <= desired_grasp_score - score < 0.1:
                                    grasp_found = True
                                    usr_confirmed = True
                                    OBJECT_SELECTED = False
                                    object_mask_save = OBJECT_MASK * 255
                                    grasp_rect = [gy + im_crop[0], gx + im_crop[2], ang]
                                    plot_grasp_path([gy, gx], ang, HH, HW, searches_show)
                                    cv2.imshow(mask_window, searches_show)
                                    break
                            grasp_tested += a
                            a = 0
                        total_time = time.time() - s_time
        usr_in = cv2.waitKey(20) & 0xff
        if usr_in == ord('q'):  # end loop
            break
        if usr_confirmed:
            print("executing grasp**********")
            print(grasp_rect)
            print(SCORE)
            # head.command_nod()
            rospy.sleep(1)
            rectangle_center = cam.rectifyPoint((grasp_rect[1], grasp_rect[0]))
            img_origin = np.array([616.24365, 422.427187, 0]).reshape((3, 1))
            center_v_img = np.array([rectangle_center[0], rectangle_center[1], 1]).reshape((3, 1)) - img_origin
            center_v_camera = np.matmul(cam_mf, center_v_img)
            center_v_camera_aug = np.append(center_v_camera, [[1]], axis=0)
            goal_in_gripper_frame = np.matmul(cam_gripper_trans, center_v_camera_aug)
            # current_pose = end_pose [x y z qx qy qz qw]
            # current_pose = dict2list(right.endpoint_pose(), ['position', 'orientation'])
            # gripper_world_trans = pose2trans(current_pose)
            goal_world = goal_in_gripper_frame
            goal_pose = geometry_msgs.msg.Pose()
            goal_pose.position = geometry_msgs.msg.Vector3(
                goal_world[0, 0] + 0.007, goal_world[1, 0] - 0.004, goal_world[2, 0])
            theta = np.deg2rad(grasp_rect[2] / 2)
            cos = np.cos(theta)
            sin = np.sin(theta)
            goal_pose.orientation = geometry_msgs.msg.Quaternion(sin, cos, 0, 0)
            print('goal pose:')
            print(goal_pose)

            usr_confirmed = False
    rospy.on_shutdown(clean_shutdown)
    cv2.destroyAllWindows()
    # # prepare tools
    # # cvbridge for converting ros image to cv image
    # br = CvBridge()
    #
    # # main loop
    # rate = rospy.Rate(10)
    # result_show_window = 'prediction results'
    # cv2.namedWindow(result_show_window, cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback(result_show_window, get_mouse_click)
    # while not rospy.is_shutdown():
    #     if INPUT_ROS_IMAGE.header.frame_id != '':
    #         # convert ros image to cv2 image
    #         in_cv_img = br.imgmsg_to_cv2(INPUT_ROS_IMAGE, 'bgr8')
    #         in_cv_img_size = (in_cv_img.shape[0], in_cv_img.shape[1])
    #         input_image = crop_image2roi(roi, in_cv_img)
    #         # print('input image size:', input_image.shape)
    #         # cv2.imshow('image', input_image)
    #         # s_time = time.time()
    #
    #         # print('mask detection takes:', time.time()-s_time, 'sec')
    #         # print('test outputs------------------------')
    #         # print(outputs['instances'])
    #         # print(predictions)
    #
    #         # print(boxes)
    #
    #         key = cv2.waitKey(5) & 0xff
    #         if MOUSE_CLICKED_AT != (-1, -1):
    #             print('Mouse clicked at: ', MOUSE_CLICKED_AT)
    #             mask_id = get_close2poi_bb_id(MOUSE_CLICKED_AT, boxes)
    #             print('Detection result id:', mask_id)
    #             if mask_id is not None:
    #                 obj_mask = pad_image2shape(masks[mask_id]*255, in_cv_img_size)
    #                 obj_bb = get_bb_in_original_image((boxes[mask_id][0], boxes[mask_id][1], boxes[mask_id][2],
    #                                                    boxes[mask_id][3]), roi, in_cv_img_size)
    #                 obj_bb_str = '{}, {}, {}, {}'.format(obj_bb[0], obj_bb[1], obj_bb[2], obj_bb[3])
    #                 print('Selected object bounding box: ', obj_bb_str)
    #                 show_mask = np.stack((obj_mask, obj_mask, obj_mask), axis=2)
    #                 cv2.rectangle(show_mask, (obj_bb[0], obj_bb[1]), (obj_bb[2], obj_bb[3]), (255, 0, 0), 2)
    #                 cv2.imshow('clicked_object_mask', show_mask)
    #                 print('mask image size: ', obj_mask.shape)
    #                 uso_mask_imgmsg = br.cv2_to_imgmsg(obj_mask)
    #             else:
    #                 uso_mask_imgmsg = sensor_msgs.msg.Image()
    #                 obj_bb_str = ''
    #             # ------------------- publish results---------------------
    #             uso_mask_pub.publish(uso_mask_imgmsg)
    #             uso_bb_pub.publish(obj_bb_str)
    #             # --------------------------------------------------------
    #             cv2.waitKey(0)
    #         if key == ord('q'):
    #             break
    #         MOUSE_CLICKED_AT = (-1, -1)
    #     rate.sleep()


if __name__ == '__main__':
    main()
