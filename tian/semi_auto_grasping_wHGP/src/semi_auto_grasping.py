import os
from cv2 import cv2
import rospy
import pyrealsense2 as realsense
import numpy as np
from pyquaternion import Quaternion
from kortex_driver.msg import *
import geometry_msgs.msg
from open3d import open3d as o3d
from sensor_msgs.msg import Image
from std_msgs.msg import Empty as std_Empty
# from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge
from utils import view_param2cart_pose, cartesian2spherical_coords, test_view_params
from tt_move_gen3 import MoveMyGen3, jacobian_spherical2cartesian
import pygame
from tt_pygame_util import user_input_confirm_or_cancel_gui
from scipy.spatial import KDTree
# os.environ["ROS_NAMESPACE"] = "/my_gen3/"
bridge = CvBridge()
# xyz = np.empty((0, 3))
# rgb = np.empty((0, 3))
hand_cam_color = np.zeros((100, 100))
hand_cam_depth = np.zeros((100, 100))


def get_current_spherical_coords(robot_control, t_0b__):
    current_p = robot_control.arm_group.get_current_pose().pose.position
    current_p_b = np.array([[current_p.x], [current_p.y], [current_p.z], [1]])
    current_p_0 = np.matmul(t_0b__, current_p_b).flatten()
    return cartesian2spherical_coords(current_p_0)


def color_callback(color_msg):
    global hand_cam_color
    hand_cam_color = bridge.imgmsg_to_cv2(color_msg, 'rgb8')


def depth_callback(depth_msg):
    global hand_cam_depth
    hand_cam_depth = bridge.imgmsg_to_cv2(depth_msg)


# def pc2_callback(ros_pc2_msg):
#     global xyz, rgb
#     cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(ros_pc2_msg)
#     rgb_arr = cloud_arr['rgb']
#     rgb_arr.dtype = np.uint32
#     r = np.asarray((rgb_arr >> 16) & 255, dtype=np.float64)/255
#     g = np.asarray((rgb_arr >> 8) & 255, dtype=np.float64)/255
#     b = np.asarray(rgb_arr & 255, dtype=np.float64)/255
#     rgb = np.stack((r, g, b), axis=2).reshape((-1, 3))
#     xyz = np.stack((cloud_arr['x'], cloud_arr['y'], cloud_arr['z']), axis=2).reshape((-1, 3))


# def save_point_cloud_to_file(file_name, pc_xyz, pc_rgb):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc_xyz)
#     pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
#     o3d.io.write_point_cloud(file_name, pcd)
#     print('number of points: ', pc_xyz.shape, pc_rgb.shape)
#     print('point cloud data saved to: ', file_name)


def main():
    # ----------------------------------ROS stuff ------------------------------------------
    rospy.init_node('TT_semi_auto_grasping', anonymous=True)
    rospy.Subscriber('/camera/color/image_rect_color', Image, color_callback, queue_size=1)
    rospy.Subscriber('/camera/depth_registered/sw_registered/image_rect', Image, depth_callback, queue_size=1)
    # rospy.Subscriber('/camera/depth_registered/points', PointCloud2, pc2_callback, queue_size=1)
    twist_command_pub = rospy.Publisher('/my_gen3/in/cartesian_velocity', kortex_driver.msg.TwistCommand, queue_size=5)
    stop_command_pub = rospy.Publisher('/my_gen3/in/stop', std_Empty, queue_size=1)
    # ------------------------------------ parameters -----------------------------------------
    rospy.loginfo('--------------Initializing parameters-------------')
    global hand_cam_color, hand_cam_depth  # xyz, rgb
    pre_d = 0.25  # in meter
    # ---------------- eye-to-hand camera pose in robot base frame -----------------
    # (0.703286572, -0.479129947, 0.407774383, Quaternion(0.446737, -0.846885, -0.243772, 0.155097))
    # position as quaternion 0 x y z
    cam_in_robot_pose_p_q = Quaternion(0.0, 0.703286572, -0.479129947, 0.407774383)
    cam_in_robot_pose_r_q = Quaternion(0.446737, -0.846885, -0.243772, 0.155097)  # rotation as quaternion w x y z
    # ----------------------------- realsense camera initialization ---------------------------
    rospy.loginfo('--------Initializing realsense L515 camera---------')
    rs_pipe = realsense.pipeline()  # camera pipeline object
    rs_profile = rs_pipe.start()  # start camera
    # get depth scale
    rs_depth_sensor = rs_profile.get_device().first_depth_sensor()
    rs_depth_scale = rs_depth_sensor.get_depth_scale()
    print(rs_profile.get_device(), 'is enabled')
    print("Camera depth image's depth scale is: ", rs_depth_scale)
    # get depth camera intrinsics
    # rs_depth_intrinsics = rs_profile.get_stream(realsense.stream.depth).as_video_stream_profile().get_intrinsics()
    rs_color_intrinsics = rs_profile.get_stream(realsense.stream.color).as_video_stream_profile().get_intrinsics()
    # align depth image to color image
    rs_align_to = realsense.stream.color
    rs_align = realsense.align(rs_align_to)
    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        rs_pipe.wait_for_frames()
    # -----------------------------MoveIt! initialization -------------------------------------
    rospy.loginfo('------Initializing Kinova gen3 arm with MoveIt-------')
    move_gen3 = MoveMyGen3()
    success = move_gen3.is_init_success

    # ----------------- user interface ---------------------
    pygame.init()
    # fps = 40
    # fpsclock = pygame.time.Clock()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("Semi-autonomous grasping GUI")
    arrow_img = pygame.image.load('arrow.png')
    font = pygame.font.SysFont('comicsans', 40)
    white = (255, 255, 255)
    rate = rospy.Rate(40)
    screen.fill(white)
    button_w = 80
    button_h = 50
    button_gap = 10
    dy = 20
    # --------------------------- add collision boxes to the planning scene -----------------------
    if success:
        rospy.loginfo('ROS node for Kinova gen3 arm with MoveIt initialization complete.')
        # -------------------------- read in scene point cloud ---------------------------
        cl_bbx_positions = np.loadtxt('bbx_positions.txt')
        cl_bbx_sizes = np.loadtxt('bbx_sizes.txt')
        for i in range(cl_bbx_positions.shape[0]):
            cb_name = 'cluster_' + str(i)
            move_gen3.add_collision_box(cb_name, cl_bbx_sizes[i], cl_bbx_positions[i])
        cl_bbx_kdtree = KDTree(cl_bbx_positions)

        move_gen3.add_collision_box('base_table', None, None)
        move_gen3.add_collision_box('back_block', (0.1, 2, 1), (-0.25, 0.0, 0.5))
        move_gen3.reach_named_position('home', 1)
        # ------------------------ local scene scanning -----------------------
        # current_dir = os.getcwd()
        # views_save_dir = os.path.join(current_dir, 'saved_views')
        rospy.loginfo('Semi autonomous grasping system ready for action.')
        # ---------- stage 1 move close to the object of interest ----------
        stage1_complete = False
        poi_in_rob = None
        rospy.loginfo('stage 1 - select object for grasping')
        iter_counter = 0
        while not (stage1_complete or rospy.is_shutdown()):  # loop 1
            user_selected_point = (-1, -1)
            # ------ get eye-to-hand camera image -----
            frames = rs_pipe.wait_for_frames()
            # Get aligned frames
            aligned_frames = rs_align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue  # back to loop 1 starting point
            body_cam_color = np.asanyarray(color_frame.get_data())
            # body_cam_color = cv2.cvtColor(body_cam_color, cv2.COLOR_RGB2BGR)
            # pygame GUI display image
            scene_img = pygame.surfarray.make_surface(body_cam_color)
            scene_img = pygame.transform.rotate(scene_img, 90)
            scene_img = pygame.transform.flip(scene_img, False, True)
            screen.blit(scene_img, (0, 0))
            if iter_counter == 0:
                rospy.loginfo('Please select the object for grasping')
            for eve in pygame.event.get():
                if eve.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if eve.type == pygame.KEYDOWN:
                    if eve.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                if eve.type == pygame.MOUSEBUTTONDOWN:
                    if eve.button == 1:
                        mx, my = pygame.mouse.get_pos()
                        print('Mouse clicked at: ', mx, my)
                        pygame.draw.circle(screen, (255, 0, 0), (mx, my), 4)
                        user_confirmed = user_input_confirm_or_cancel_gui(screen, button_w, button_h,
                                                                          mx, my, dy, button_gap)
                        if user_confirmed:
                            user_selected_point = (mx, my)

            # ----- user select object -----
            if user_selected_point != (-1, -1):
                rospy.loginfo('--------------------------------------')
                print('mouse clicked at: ', user_selected_point)
                distance = aligned_depth_frame.get_distance(user_selected_point[0], user_selected_point[1])
                print('The depth at clicked point is: ', distance)
                poi = [float(user_selected_point[0]), float(user_selected_point[1])]
                poi_in_cam = realsense.rs2_deproject_pixel_to_point(rs_color_intrinsics, poi, distance)
                print('Clicked 3d location relative to camera: ', poi_in_cam)
                poi_in_cam_q = Quaternion(0.0, poi_in_cam[0], poi_in_cam[1], poi_in_cam[2] + 0.03)
                print('Clicked 3d location relative to camera as quaternion: ', poi_in_cam_q)
                cam_in_rob_pose_r_q_conj = cam_in_robot_pose_r_q.conjugate
                poi_in_cam_in_rob = cam_in_robot_pose_r_q * poi_in_cam_q * cam_in_rob_pose_r_q_conj
                poi_in_rob = cam_in_robot_pose_p_q + poi_in_cam_in_rob
                print('Clicked point 3D location relative to robot: ', (poi_in_rob.x, poi_in_rob.y, poi_in_rob.z))
                # --------------- find closest cluster ---------------------
                dist, ind = cl_bbx_kdtree.query([poi_in_rob.x, poi_in_rob.y, poi_in_rob.z])
                print('Selected cluster index: ', ind, 'with distance: ', dist)
                refined_selection = cl_bbx_kdtree.data[ind]
                cl_file = 'cluster' + str(ind) + '.pcd'
                cl_points = o3d.io.read_point_cloud(cl_file)
                o3d.visualization.draw_geometries([cl_points], window_name='User selected object point cloud')
                poi_in_rob.x = refined_selection[0]
                poi_in_rob.y = refined_selection[1]
                poi_in_rob.z = refined_selection[2]
                # ----- go to default initial pre-grasp pose -----
                default_pose_params = [[np.pi, 0.0, pre_d],
                                       [np.pi, np.pi / 2, pre_d],
                                       [1.25 * np.pi, np.pi / 2, pre_d],
                                       [0.75 * np.pi, np.pi / 2, pre_d],
                                       [np.pi, np.pi / 4, pre_d],
                                       [1.25 * np.pi, np.pi / 4, pre_d],
                                       [0.75 * np.pi, np.pi / 4, pre_d]]

                for count, pose_param in enumerate(default_pose_params, 1):  # loop 2
                    # ------------------------- GUI update
                    screen.blit(scene_img, (0, 0))
                    display_str = "Searching for reachable default pre-grasp pose: " + str(count) + "/5"
                    text = font.render(display_str, True, (200, 200, 0))
                    screen.blit(text, (280, 280))
                    pygame.display.update()
                    # ------------------------------------
                    # print('Testing default pre-grasp pose reachability: ', count, pose_param)
                    default_pose = view_param2cart_pose(poi_in_rob, pose_param)
                    reachable, plan = move_gen3.plan_to_cartesian_pose(default_pose, 0.5)
                    if reachable:
                        # ------------------------- GUI update
                        screen.blit(scene_img, (0, 0))
                        text = font.render("Valid default pre-grasp pose found, "
                                           "check robot trajectory and confirm execution.",
                                           True, (200, 200, 0))
                        screen.blit(text, (100, 280))
                        user_confirmed = user_input_confirm_or_cancel_gui(screen, button_w, button_h,
                                                                          640, 380, -button_h / 2, button_gap)
                        # ------------------------------------
                        if user_confirmed:
                            # ------------------------- GUI update
                            screen.blit(scene_img, (0, 0))
                            text = font.render("Robot moving to default pre-grasp pose",
                                               True, (200, 200, 0))
                            screen.blit(text, (400, 280))
                            pygame.display.update()
                            # ------------------------------------
                            stage1_complete = move_gen3.execute_trajectory_plan(plan)
                            if stage1_complete:
                                break  # exit loop 2
                iter_counter = 0
            else:
                pygame.display.update()
                iter_counter += 1
            rate.sleep()
        # !----!!!!!--- --------------------------------------------!!!!!----!
        # ---------- stage 2 close up scan ----------
        rospy.loginfo('stage 2 - Grasp direction adjustment')
        # ------------- robot velocity control parameters
        # print(poi_in_rob)
        x0 = poi_in_rob.x
        y0 = poi_in_rob.y
        z0 = poi_in_rob.z
        # print(x0, y0, z0)
        n_xy0 = (x0 ** 2 + y0 ** 2) ** 0.5
        c0 = x0 / n_xy0
        s0 = y0 / n_xy0
        t_0b = np.array([[c0, s0, 0, -(x0 * c0 + y0 * s0)],
                         [-s0, c0, 0, x0 * s0 - y0 * c0],
                         [0, 0, 1, -z0],
                         [0, 0, 0, 1]])
        twist_command = kortex_driver.msg.TwistCommand()
        twist_command.reference_frame = 3
        stop_command = std_Empty()

        # --------------GUI
        screen.fill(white)
        text = font.render("Use arrow keys to adjust grasp direction. Press e to confirm.",
                           True, (0, 0, 200))
        screen.blit(text, (270, 70))
        hand_cam_color_dx = 320
        hand_cam_color_dy = 120
        # ----------------
        stage2_complete = False
        while not (rospy.is_shutdown() or stage2_complete):
            # ------------------ default velocity ----------------------
            phi_dot = 0.0
            theta_dot = 0.0
            stop = True
            # ------------------ GUI
            # hand camera view
            color_display_img = pygame.surfarray.make_surface(hand_cam_color)
            color_display_img = pygame.transform.rotate(color_display_img, 90)
            color_display_img = pygame.transform.flip(color_display_img, False, True)
            screen.blit(color_display_img, (hand_cam_color_dx, hand_cam_color_dy))
            up_arrow_img = arrow_img.copy()
            key_input = pygame.key.get_pressed()
            # -----------------------
            unit_angular_velocity = 4 * np.pi / 180
            for eve in pygame.event.get():
                if eve.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if key_input[pygame.K_LEFT]:
                phi_dot = -unit_angular_velocity
                left_arrow_img = pygame.transform.rotate(up_arrow_img, 90)
                screen.blit(left_arrow_img, (hand_cam_color_dx, 190 + hand_cam_color_dy))
                stop = False
            if key_input[pygame.K_UP]:
                theta_dot = unit_angular_velocity
                screen.blit(up_arrow_img, (270 + hand_cam_color_dx, hand_cam_color_dy))
                stop = False
            if key_input[pygame.K_RIGHT]:
                phi_dot = unit_angular_velocity
                right_arrow_img = pygame.transform.rotate(up_arrow_img, -90)
                screen.blit(right_arrow_img, (520 + hand_cam_color_dx, 190 + hand_cam_color_dy))
                stop = False
            if key_input[pygame.K_DOWN]:
                theta_dot = -unit_angular_velocity
                down_arrow_img = pygame.transform.rotate(up_arrow_img, 180)
                screen.blit(down_arrow_img, (270 + hand_cam_color_dx, 360 + hand_cam_color_dy))
                stop = False
            if key_input[pygame.K_ESCAPE]:
                pygame.quit()
                sys.exit()
            if key_input[pygame.K_e]:
                stage2_complete = True
            pygame.display.update()
            # --------------------------------------------------------------
            # ---------------------- calculate velocity command -----------------------
            if stop:
                stop_command_pub.publish(stop_command)
            else:
                current_spherical_coords = get_current_spherical_coords(move_gen3, t_0b)
                print(current_spherical_coords)
                # print([current_p.x, current_p.y, current_p.z, current_pose.orientation.x, current_pose.orientation.y,
                #        current_pose.orientation.z, current_pose.orientation.w])
                j__63 = jacobian_spherical2cartesian(c0, s0, current_spherical_coords, angle_representation='radians')
                twist__61 = np.matmul(j__63, np.array([[phi_dot], [theta_dot], [0.0]]))
                # print('velocity command')
                twist_command.twist.linear_x = twist__61[0, 0]
                twist_command.twist.linear_y = twist__61[1, 0]
                twist_command.twist.linear_z = twist__61[2, 0]
                twist_command.twist.angular_x = twist__61[3, 0]
                twist_command.twist.angular_y = twist__61[4, 0]
                twist_command.twist.angular_z = twist__61[5, 0]
                # print(twist_command)
                twist_command_pub.publish(twist_command)
            rate.sleep()

        if stage2_complete:
            rospy.loginfo('Stage 3 - Object silhouette extractioin and neural network grasp detection')
            # 1. Transform object point cloud to hand camera coordinate frame
            # 2. Extract object silhouette based on grasp depth
            # 3. Grasp detection using object silhouette

            # # get current robot pose parameters
            # current_spherical_coords = get_current_spherical_coords(move_gen3, t_0b)
            # rospy.loginfo('Current robot position in object spherical coordinate system:')
            # print(current_spherical_coords)
            # pre_grasp_pose_params = np.copy(current_spherical_coords)
            # # 1. Find scanning view poses
            # # left-most view
            # d_phi_left = [-1.047, -0.523]  # 60 , 30
            # if pre_grasp_pose_params[1] < 0.8:  # 45
            #     thetas = [1.134, 0.96]
            # else:
            #     thetas = [pre_grasp_pose_params[1]]
            #
            # rospy.loginfo('-----------view pose determination-----------')
            # print(pre_grasp_pose_params)
            # print(thetas)
            # saved_view_poses = []
            # rospy.loginfo('left side:')
            # view_pose_reached = False
            # for theta_i in thetas:
            #     if view_pose_reached:
            #         break
            #     for d_phi in d_phi_left:
            #         view_params = [pre_grasp_pose_params[0] + d_phi, theta_i, pre_d]
            #         print(view_params)
            #         view_pose_reached, saved_view_poses = test_view_params(poi_in_rob, view_params, move_gen3,
            #                                                                saved_view_poses,
            #                                                                hand_cam_color, hand_cam_depth,
            #                                                                '1.png', '1_.png')
            #         if view_pose_reached:
            #             break
            #
            # # middle view
            # # right-most view
            # rospy.loginfo('right side:')
            # d_phi_right = [1.047, 0.523]  # 60 , 30
            # view_pose_reached = False
            # for theta_i in thetas:
            #     if view_pose_reached:
            #         break
            #     for d_phi in d_phi_right:
            #         view_params = [pre_grasp_pose_params[0] + d_phi, theta_i, pre_d]
            #         print(view_params)
            #         view_pose_reached, saved_view_poses = test_view_params(poi_in_rob, view_params, move_gen3,
            #                                                                saved_view_poses,
            #                                                                hand_cam_color, hand_cam_depth,
            #                                                                '2.png', '2_.png')
            #         if view_pose_reached:
            #             break
            # # top view
            # rospy.loginfo('top side:')
            # # check if pre-grasp pose is close to top scan view pose
            # if pre_grasp_pose_params[1] > 0.174:
            #     up_theta = 0.0
            # else:
            #     up_theta = pre_grasp_pose_params[1]
            #     print('pre-grasp pose close to top view pose')
            # while up_theta < pre_grasp_pose_params[1]:
            #     view_params = [pre_grasp_pose_params[0], up_theta, pre_d]
            #     print(view_params)
            #     view_pose_reached, saved_view_poses = test_view_params(poi_in_rob, view_params, move_gen3,
            #                                                            saved_view_poses,
            #                                                            hand_cam_color, hand_cam_depth,
            #                                                            '3.png', '3_.png')
            #     if view_pose_reached:
            #         break
            #     else:
            #         up_theta += 0.174
            #
            # rospy.loginfo('back to pre-grasp view')
            # view_pose_reached, saved_view_poses = test_view_params(poi_in_rob, pre_grasp_pose_params, move_gen3,
            #                                                        saved_view_poses,
            #                                                        hand_cam_color, hand_cam_depth,
            #                                                        '4.png', '4_.png',
            #                                                        robot_pose_tolerance=(0.01, 0.04))
            # np.savetxt('view_poses.txt', saved_view_poses, fmt='%1.4f')
            # if view_pose_reached:
            #     rospy.loginfo('Scanning phase complete.')
            #     rospy.loginfo('Waiting for object silhouette extraction and initial grasp pose detection...')

            # 3. Object point cloud registration and silhouette extraction (another script)
            # 4. Neural network grasp detection (another script)


if __name__ == "__main__":
    main()
