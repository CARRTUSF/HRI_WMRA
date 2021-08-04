import os
import sys
import rospy
import cv2
import moveit_commander
import ros_numpy
import pyrealsense2 as realsense
import open3d as o3d
import numpy as np
# import quaternion   # numpy-quaternion
from pyquaternion import Quaternion
import moveit_msgs.msg
import geometry_msgs.msg
# from moveit_commander.conversions import pose_to_list
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge
from utils import view_param2cart_pose, cartesian2spherical_coords

os.environ["ROS_NAMESPACE"] = "/my_gen3/"
bridge = CvBridge()
SCENE_IMG_CLICKED_AT = (-1, -1)
# xyz = np.empty((0, 3))
# rgb = np.empty((0, 3))
color_img = np.zeros((100, 100))
depth_img = np.zeros((100, 100))


def get_quaternion_pose_from_xyz_axes(x_axis, y_axis, z_axis):
    rotation_matrix = np.array([[x_axis[0], y_axis[0], z_axis[0]],
                                [x_axis[1], y_axis[1], z_axis[1]],
                                [x_axis[2], y_axis[2], z_axis[2]]])
    pose_quaternion = Quaternion(matrix=rotation_matrix)
    return pose_quaternion


def ros_pose_from_trans_matrix(_trans_matrix):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = _trans_matrix[0, 3]
    pose.position.y = _trans_matrix[1, 3]
    pose.position.z = _trans_matrix[2, 3]
    pose.orientation = Quaternion(matrix=_trans_matrix)
    return pose


def generate_scanning_waypoints(_poi, waypoints_params):
    waypoints = []
    n_xy = (_poi.x**2 + _poi.y**2)**0.5
    cos_t = _poi.x / n_xy
    sin_t = _poi.y / n_xy
    # print(cos_t, sin_t)
    T_bo = np.zeros((4, 4))
    T_bo[0, 0] = cos_t
    T_bo[0, 1] = -sin_t
    T_bo[0, 3] = _poi.x
    T_bo[1, 0] = sin_t
    T_bo[1, 1] = cos_t
    T_bo[1, 3] = _poi.y
    T_bo[2, 2] = 1
    T_bo[2, 3] = _poi.z
    T_bo[3, 3] = 1
    # print(T_bo)
    for i in range(waypoints_params.shape[0]):
        p_ix = np.cos(waypoints_params[i, 0]) * np.sin(waypoints_params[i, 1]) * waypoints_params[i, 2]
        p_iy = np.sin(waypoints_params[i, 0]) * np.sin(waypoints_params[i, 1]) * waypoints_params[i, 2]
        p_iz = np.cos(waypoints_params[i, 1]) * waypoints_params[i, 2]
        n_pi = (p_ix**2 + p_iy**2 + p_iz**2)**0.5
        vz_i = - np.array([p_ix, p_iy, p_iz]) / n_pi
        vx_i = np.array([0.0, 1.0, 0.0])
        if vz_i[0] != 0.0 or vz_i[1] != 0.0:
            n_xxy = (vz_i[0]**2 + vz_i[1]**2)**0.5
            vx_i[0] = - vz_i[1] / n_xxy
            vx_i[1] = vz_i[0] / n_xxy
            vx_i[2] = 0.0
        vy_i = np.cross(vz_i, vx_i)
        T_oi = np.zeros((4, 4))
        T_oi[:3, 0] = vx_i
        T_oi[:3, 1] = vy_i
        T_oi[:3, 2] = vz_i
        T_oi[0, 3] = p_ix
        T_oi[1, 3] = p_iy
        T_oi[2, 3] = p_iz
        T_oi[3, 3] = 1.0
        T_bi = np.matmul(T_bo, T_oi)
        # print('[][][][][][]')
        # print(T_oi)
        # print(T_bi)
        pose_i = ros_pose_from_trans_matrix(T_bi)
        waypoints.append(pose_i)
    return waypoints


def scene_img_click(event, x, y, flags, param):
    global SCENE_IMG_CLICKED_AT
    if event == cv2.EVENT_LBUTTONDOWN:
        SCENE_IMG_CLICKED_AT = (x, y)


def color_callback(color_msg):
    global color_img
    color_img = bridge.imgmsg_to_cv2(color_msg, 'bgr8')


def depth_callback(depth_msg):
    global depth_img
    depth_img = bridge.imgmsg_to_cv2(depth_msg)


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


def save_point_cloud_to_file(file_name, pc_xyz, pc_rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    o3d.io.write_point_cloud(file_name, pcd)
    print('number of points: ', pc_xyz.shape, pc_rgb.shape)
    print('point cloud data saved to: ', file_name)


def main():
    # ----------------------------------ROS stuff ------------------------------------------
    rospy.init_node('TT_semi_auto_grasping', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, color_callback, queue_size=1)
    rospy.Subscriber('/camera/depth_registered/sw_registered/image_rect', Image, depth_callback, queue_size=1)
    # rospy.Subscriber('/camera/depth_registered/points', PointCloud2, pc2_callback, queue_size=1)
    # ------------------------------------ parameters -----------------------------------------
    print('--------------Initializing parameters-------------')
    global SCENE_IMG_CLICKED_AT, color_img, depth_img  # xyz, rgb
    pre_d = 0.25    # in meter
    # ---------------- eye-to-hand camera pose in robot base frame -----------------
    # (0.703286572, -0.479129947, 0.407774383, Quaternion(0.446737, -0.846885, -0.243772, 0.155097))
    # position as quaternion 0 x y z
    cam_in_robot_pose_p_q = Quaternion(0.0, 0.703286572, -0.479129947, 0.407774383)
    cam_in_robot_pose_r_q = Quaternion(0.446737, -0.846885, -0.243772, 0.155097)  # rotation as quaternion w x y z

    # ----------------------------- realsense camera initialization ---------------------------
    print('--------initializing realsense L515 camera---------')
    rs_pipe = realsense.pipeline()  # camera pipeline object
    rs_profile = rs_pipe.start()  # start camera
    # get depth scale
    rs_depth_sensor = rs_profile.get_device().first_depth_sensor()
    rs_depth_scale = rs_depth_sensor.get_depth_scale()
    print rs_profile.get_device(), 'is enabled'
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
    print('---------------Initializing MoveIt-------------')
    moveit_commander.roscpp_initialize(sys.argv)
    is_gripper_present = rospy.get_param("/my_gen3/" + "is_gripper_present", False)
    if is_gripper_present:
        gripper_joint_names = rospy.get_param("/my_gen3/" + "gripper_joint_names", [])
        gripper_joint_name = gripper_joint_names[0]
    else:
        gripper_joint_name = ""
        degrees_of_freedom = rospy.get_param("/my_gen3/" + "degrees_of_freedom", 7)
    # Create the MoveItInterface necessary objects
    arm_group_name = "arm"
    robot = moveit_commander.RobotCommander("robot_description")
    scene = moveit_commander.PlanningSceneInterface(ns="/my_gen3/")
    arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns="/my_gen3/")
    display_trajectory_publisher = rospy.Publisher("/my_gen3/" + 'move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)
    if is_gripper_present:
        gripper_group_name = "gripper"
        gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns="/my_gen3/")
    rospy.loginfo("Initializing node in namespace " + "/my_gen3/")
    # --------------------------- add collision boxes to the planning scene -----------------------
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = "base_link"
    box_pose.pose.orientation.w = 1.0
    box_pose.pose.position.x = 0.4
    box_pose.pose.position.y = 0.0
    box_pose.pose.position.z = -0.2
    scene.add_box('table', box_pose, size=(1, 2, 0.4))
    # ------------------------ local scene scanning -----------------------
    current_dir = os.getcwd()
    views_save_dir = os.path.join(current_dir, 'saved_views')
    print('-----------main loop started-------------')
    quit_ = False
    while not rospy.is_shutdown():
        if quit_:
            break
        scene_img_window = 'Scene_image'
        cv2.namedWindow(scene_img_window)
        cv2.setMouseCallback(scene_img_window, scene_img_click)
        # moved2object = False
        poi_in_rob = None
        # ---------------------------stage 1 move close to the object of interest ----------------------
        stage1_complete = False
        while not stage1_complete:     # wait for the user to select the object of interest
            frames = rs_pipe.wait_for_frames()
            # Get aligned frames
            aligned_frames = rs_align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            # depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(scene_img_window, color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                quit_ = True
                break
            if SCENE_IMG_CLICKED_AT != (-1, -1):
                print '--------------------------------------'
                print 'mouse clicked at: ', SCENE_IMG_CLICKED_AT
                distance = aligned_depth_frame.get_distance(SCENE_IMG_CLICKED_AT[0], SCENE_IMG_CLICKED_AT[1])
                print 'The depth at clicked point is: ', distance
                poi = [float(SCENE_IMG_CLICKED_AT[0]), float(SCENE_IMG_CLICKED_AT[1])]
                poi_in_cam = realsense.rs2_deproject_pixel_to_point(rs_color_intrinsics, poi, distance)
                print 'Clicked 3d location relative to camera: ', poi_in_cam
                poi_in_cam_q = Quaternion(0.0, poi_in_cam[0], poi_in_cam[1], poi_in_cam[2])
                print 'Clicked 3d location relative to camera as quaternion: ', poi_in_cam_q
                cam_in_rob_pose_r_q_conj = cam_in_robot_pose_r_q.conjugate
                poi_in_cam_in_rob = cam_in_robot_pose_r_q * poi_in_cam_q * cam_in_rob_pose_r_q_conj
                poi_in_rob = cam_in_robot_pose_p_q + poi_in_cam_in_rob
                print 'Clicked point 3D locatioin relative to robot: ', (poi_in_rob.x, poi_in_rob.y, poi_in_rob.z)
                # !----!!!!!--- replace this section with user controlled grasp orientation selection ---!!!!!----!
                # go to default initial pre-grasp pose
                default_pose_params = [[np.pi, np.pi/4, pre_d],
                                       [np.pi, np.pi/2, pre_d],
                                       [0.75*np.pi, np.pi/2, pre_d],
                                       [1.25*np.pi, np.pi/2, pre_d],
                                       [1.25*np.pi, np.pi/4, pre_d]]
                arm_group.set_max_velocity_scaling_factor(0.5)
                arm_group.set_goal_position_tolerance(0.01)
                arm_group.set_goal_orientation_tolerance(0.1)
                auto_reach_complete = False
                while not auto_reach_complete:
                    for pose_param in default_pose_params:
                        default_pose = view_param2cart_pose(poi_in_rob, pose_param)
                        arm_group.set_pose_target(default_pose)
                        plan = arm_group.plan()
                        if plan.joint_trajectory.points:
                            user_confirm = raw_input("execute planed trajectory? (y/n)")
                            if user_confirm == 'y':
                                arm_group.execute(plan, wait=True)
                                arm_group.stop()
                                arm_group.clear_pose_targets()
                                auto_reach_complete = True
                                break

                if auto_reach_complete:
                    user_adjust_complete = False
                    # user adjust grasp direction
                    while not user_adjust_complete:



                # !----!!!!!--- --------------------------------------------------------------------- ---!!!!!----!
        # -------------------------------------- stage 2 close up scan ---------------------------------
        cv2.destroyAllWindows()
        print(poi_in_rob)
        raw_input('Waiting for grasp direction adjustment...')
        if poi_in_rob is not None and False:
            upside_thetas = [0.0, 0.174, 0.348, 0.522]
            downside_thetas = [1.57, 1.39, 1.22]
            phis_1 = [1.74, 2.356]
            phis_2 = [4.53, 3.927]
            phis = [1.74, 2.356, np.pi, 4.53, 3.927]
            current_pose = arm_group.get_current_pose().pose
            current_pose_param = cartesian2spherical_coords(current_pose.position)

            # up side view pose
            for theta_i in upside_thetas:
                view_params = [np.pi, theta_i, pre_d]
                view_pose_ij = view_param2cart_pose(poi_in_rob, view_params)
                reachability_ij, plan = move_my_gen3.plan_to_cartesian_pose(view_pose_ij, 0.5)
                if reachability_ij == 1:
                    print('============')
                    print(view_params)
                    trajectory_params = scanning_trajectory_waypoint_poses(
                        [[np.pi, np.pi / 4, pre_d], [np.pi, theta_i, pre_d]])
                    reverse_trajectory_params = scanning_trajectory_waypoint_poses(
                        [[np.pi, theta_i, pre_d], [np.pi, np.pi / 4, pre_d]])
                    print(trajectory_params)
                    print(reverse_trajectory_params)
                    trajectory_poses = generate_scanning_waypoints(poi_in_rob, trajectory_params)
                    reverse_trajectory_poses = generate_scanning_waypoints(poi_in_rob, reverse_trajectory_params)
                    print(trajectory_poses)
                    print(reverse_trajectory_poses)
                    trajectory_plan, trajectory_frac = move_my_gen3.plan_trajectory_from_waypoints(trajectory_poses,
                                                                                                   display_trajectory=True)
                    # print(trajectory_plan)
                    print(trajectory_frac)
                    if trajectory_frac > 0.7:
                        move_my_gen3.execute_trajectory_plan(trajectory_plan, wait=True)
                        # take images
                        rospy.sleep(2)
                        reverse_trajectory_poses[0] = move_my_gen3.arm_group.get_current_pose().pose
                        print(reverse_trajectory_poses)
                        reverse_trajectory_plan, reverse_trajectory_frac = move_my_gen3.plan_trajectory_from_waypoints(
                            reverse_trajectory_poses,
                            display_trajectory=True)
                        print(reverse_trajectory_frac)
                        if reverse_trajectory_frac > 0.7:
                            move_my_gen3.execute_trajectory_plan(reverse_trajectory_plan, wait=True)
                    break

            # down side view poses
            view_pose_found = False
            for phi in phis_1:
                if not view_pose_found:
                    for theta_i in downside_thetas:
                        view_params = [phi, theta_i, pre_d]
                        print(view_params)
                        view_pose_ij = view_param2cart_pose(poi_in_rob, view_params)
                        reachability_ij, plan = move_my_gen3.plan_to_cartesian_pose(view_pose_ij, 1)
                        if reachability_ij == 1:
                            view_pose_found = True
                            print('============')
                            print(view_params)
                            current_p = move_my_gen3.arm_group.get_current_pose().pose.position
                            current_p_ = [current_p.x - poi_in_rob.x, current_p.y - poi_in_rob.y,
                                          current_p.z - poi_in_rob.z]
                            current_pose_param = cartesian2spherical_coords(current_p_)
                            print(current_pose_param)
                            trajectory_params = scanning_trajectory_waypoint_poses([current_pose_param, view_params])
                            print(trajectory_params)
                            reverse_trajectory_params = scanning_trajectory_waypoint_poses(
                                [view_params, [np.pi, np.pi / 4, pre_d]])
                            trajectory_poses = generate_scanning_waypoints(poi_in_rob, trajectory_params)
                            reverse_trajectory_poses = generate_scanning_waypoints(poi_in_rob,
                                                                                   reverse_trajectory_params)
                            trajectory_plan, trajectory_frac = move_my_gen3.plan_trajectory_from_waypoints(
                                trajectory_poses,
                                display_trajectory=True)
                            # print(trajectory_plan)
                            print(trajectory_frac)
                            if trajectory_frac > 0.7:
                                move_my_gen3.execute_trajectory_plan(trajectory_plan, wait=True)
                                # take images
                                rospy.sleep(2)
                                reverse_trajectory_poses[0] = move_my_gen3.arm_group.get_current_pose().pose
                                print(reverse_trajectory_poses)
                                reverse_trajectory_plan, reverse_trajectory_frac = move_my_gen3.plan_trajectory_from_waypoints(
                                    reverse_trajectory_poses,
                                    display_trajectory=True)
                                print(reverse_trajectory_frac)
                                if reverse_trajectory_frac > 0.7:
                                    move_my_gen3.execute_trajectory_plan(reverse_trajectory_plan, wait=True)
                            break

                current_pose = arm_group.get_current_pose()
                current_pose_array = np.array([current_pose.position.x,
                                               current_pose.position.y,
                                               current_pose.position.z,
                                               current_pose.orientation.w,
                                               current_pose.orientation.x,
                                               current_pose.orientation.y,
                                               current_pose.orientation.z])
                cv2.imshow('hand camera color', color_img)
                cv2.imshow('hand camera depth', depth_img)
                cv2.imwrite(os.path.join(views_save_dir, 'color', str(j)+'.jpg'), color_img)
                cv2.imwrite(os.path.join(views_save_dir, 'depth', str(j) + '.png'), depth_img)
                np.savetxt(os.path.join(views_save_dir, str(j) + '_pose.txt'), current_pose_array, fmt='%.6f')
                cv2.waitKey(1)
            cv2.destroyAllWindows()
        # if moved2object:
        #     pc_file_index = -1
        #     while True:
        #         cv2.imshow('hand camera color', color_img)
        #         cv2.imshow('hand camera depth', depth_img)
        #         key = cv2.waitKey(1) & 0xff
        #         if key == ord('q'):
        #             break
        #         if key == ord('s'):
        #             pc_file_index += 1
        #             pc_file = 'pcd' + str(pc_file_index) + '.ply'
        #             save_point_cloud_to_file(pc_file, xyz, rgb)
        #     cv2.destroyAllWindows()
        #     if pc_file_index != -1:
        #         for i in range(pc_file_index+1):
        #             pc_file = 'pcd' + str(i) + '.ply'
        #             pcd = o3d.io.read_point_cloud(pc_file)
        #             o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
