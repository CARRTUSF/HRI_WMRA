import rospy
import geometry_msgs.msg
import numpy as np
from cv2 import cv2
from pyquaternion import Quaternion


def cartesian2spherical_coords(x_):
    r = (x_[0] ** 2 + x_[1] ** 2 + x_[2] ** 2) ** 0.5
    theta = np.arccos(x_[2] / r)
    sr = np.sin(theta) * r
    if sr != 0.0:
        phi = np.arctan2(x_[1] / sr, x_[0] / sr)
    else:
        phi = 0.0
    return [phi, theta, r]


def ros_pose_from_trans_matrix(_trans_matrix):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = _trans_matrix[0, 3]
    pose.position.y = _trans_matrix[1, 3]
    pose.position.z = _trans_matrix[2, 3]
    pose.orientation = Quaternion(matrix=_trans_matrix)
    return pose


def view_param2cart_pose(_poi, view_params):
    n_xy = (_poi.x ** 2 + _poi.y ** 2) ** 0.5
    cos_t = _poi.x / n_xy
    sin_t = _poi.y / n_xy
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
    px_oi = np.cos(view_params[0]) * np.sin(view_params[1]) * view_params[2]
    py_oi = np.sin(view_params[0]) * np.sin(view_params[1]) * view_params[2]
    pz_oi = np.cos(view_params[1]) * view_params[2]
    n_p = (px_oi ** 2 + py_oi ** 2 + pz_oi ** 2) ** 0.5
    za_oi = - np.array([px_oi, py_oi, pz_oi]) / n_p
    xa_oi = np.array([0.0, 1.0, 0.0])
    if za_oi[0] != 0.0 or za_oi[1] != 0.0:
        n_xxy = (za_oi[0] ** 2 + za_oi[1] ** 2) ** 0.5
        xa_oi[0] = - za_oi[1] / n_xxy
        xa_oi[1] = za_oi[0] / n_xxy
        xa_oi[2] = 0.0
    ya_oi = np.cross(za_oi, xa_oi)
    T_oi = np.zeros((4, 4))
    T_oi[:3, 0] = xa_oi
    T_oi[:3, 1] = ya_oi
    T_oi[:3, 2] = za_oi
    T_oi[0, 3] = px_oi
    T_oi[1, 3] = py_oi
    T_oi[2, 3] = pz_oi
    T_oi[3, 3] = 1.0
    T_bi = np.matmul(T_bo, T_oi)
    pose_i = ros_pose_from_trans_matrix(T_bi)
    return pose_i


def augment_waypoints(current_waypoints, n2add):
    n = len(current_waypoints)
    augmented_waypoints = np.empty((n + n2add*(n - 1), 3))
    n_plus = n2add + 1
    for i in range(n - 1):
        step_i = (np.array(current_waypoints[i + 1]) - np.array(current_waypoints[i])) / n_plus
        for j in range(n_plus):
            augmented_waypoints[n_plus * i + j] = np.array(current_waypoints[i]) + j * step_i
    augmented_waypoints[-1] = np.array(current_waypoints[-1])
    return augmented_waypoints


def test_view_params(poi_, view_params, robot_control, saved_view_poses, color_image, depth_image,
                     color_name, depth_name, robot_velocity_scale=0.8, robot_pose_tolerance=(0.05, 0.1)):

    view_pose = view_param2cart_pose(poi_, view_params)
    reachable, plan = robot_control.plan_to_cartesian_pose(view_pose, robot_velocity_scale, robot_pose_tolerance)
    if reachable:
        user_confirmation = raw_input('Go to view pose [y]/n ?')
        if user_confirmation == 'n':
            return False, saved_view_poses
        else:
            goal_pose_reached = robot_control.execute_trajectory_plan(plan)
            # take RGBD images after reaching the view pose
            if goal_pose_reached:
                rospy.sleep(0.5)
                current_pose = robot_control.get_current_pose_as_list()
                saved_view_poses.append(current_pose)
                cv2.imwrite(color_name, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(depth_name, depth_image)
                rospy.loginfo('View images captured')
                return True, saved_view_poses
            else:
                return False, saved_view_poses
    else:
        return False, saved_view_poses
# if __name__ == '__main__':
#     wps = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
#     print(add_more_waypoints(wps, 3))
#     for nn in range(5):
#         print("--")
#         for iadd in range(5):
#             print(nn+iadd*(nn-1))
#     params = np.array([[-2.4310485955139813, 0.26037526976597325, 0.2507923407752528],
#                        [-2.349539564678498, 1.3719682708964414, 0.25911440758340376],
#                        [-1.8985077168523998, 1.3754139255946642, 0.2600017838031923],
#                        [-1.893674041130887, 0.9330255736499021, 0.25779714740652954]])
#     poi_test = Quaternion(0.0, 0.5341304794030389, 0.019225919183092965, 0.023689984066364078)
#     time1 = time.time()
#     poses = generate_scanning_poses(poi_test, params)
#     time2 = time.time()
#     poses_v1 = generate_scanning_poses_v1(poi_test, params)
#     time3 = time.time()
#     print(poses)
#     print('=====================================')
#     print(poses_v1)
#     print(time2-time1)
#     print(time3-time2)

# (0.5341304794030389, 0.019225919183092965, 0.023689984066364078)
# [-2.4310485955139813, 0.26037526976597325, 0.2507923407752528]

# [0.48673642604423195, -0.024619429220618463, 0.26602897663897973,
# 0.3512372994990512, 0.9231304355936002, 0.1334412096235129, 0.08158432383490147]
#
# [-2.349539564678498, 1.3719682708964414, 0.25911440758340376]
# [0.36233712114160505, -0.16787801242862804, 0.0748704202699889,
# 0.2935405657455727, 0.7088762852090557, 0.5735528392887341, 0.2869938833386716]
#
# [-1.8985077168523998, 1.3754139255946642, 0.2600017838031923]
# [0.4607737447932768, -0.225052142041001, 0.07416716577089941,
# 0.1219197553608933, 0.7523393928587042, 0.6286863527813717, 0.15451369205411405]
#
# [-1.893674041130887, 0.9330255736499021, 0.25779714740652954]
# [0.47551971302987256, -0.1794289516389705, 0.17718392392968013,
# 0.12371127166548512, 0.8728348388652346, 0.45547552174804457, 0.12408430365826219]
