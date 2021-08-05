import os
import sys
import copy
import geometry_msgs.msg
import rospy
import cv2
import warnings
import numpy as np
import quaternion
import moveit_commander
import moveit_msgs.msg
from pyquaternion import Quaternion

os.environ["ROS_NAMESPACE"] = "/my_gen3/"


class MoveMyGen3(object):

    def __init__(self):
        # Initialize the node
        super(MoveMyGen3, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('kinova_gen3_moveit')
        try:
            self.is_gripper_present = rospy.get_param("/my_gen3/" + "is_gripper_present", False)
            print('Is gripper present:', self.is_gripper_present)
            if self.is_gripper_present:
                gripper_joint_names = rospy.get_param("/my_gen3/" + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                gripper_joint_name = ""
            self.degrees_of_freedom = rospy.get_param("/my_gen3/" + "degrees_of_freedom", 7)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns="/my_gen3/")
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns="/my_gen3/")
            self.display_trajectory_publisher = rospy.Publisher(
                "/my_gen3/" + 'move_group/display_planned_path',
                moveit_msgs.msg.DisplayTrajectory,
                queue_size=20)

            if self.is_gripper_present:
                print('Initializing gripper move group')
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns="/my_gen3/")

            rospy.loginfo("Initializing node in namespace " + "/my_gen3/")
        except Exception as e:
            print(e)
            self.is_init_success = False
        else:
            self.is_init_success = True

    def reach_named_position(self, target, velocity_scale=0.3):
        arm_group = self.arm_group
        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        arm_group.set_named_target(target)
        # set velocity
        arm_group.set_max_velocity_scaling_factor(velocity_scale)
        # Plan the trajectory
        planned_path1 = arm_group.plan()
        # Execute the trajectory and block while it's not finished
        return arm_group.execute(planned_path1, wait=True)

    def reach_cartesian_pose(self, pose, tolerance=0.008, constraints=None, velocity_scale=0.3):
        arm_group = self.arm_group
        # Set the tolerance
        arm_group.set_goal_position_tolerance(tolerance)
        # Set the trajectory constraint if one is specified
        if constraints is not None:
            arm_group.set_path_constraints(constraints)
        # Get the current Cartesian Position
        arm_group.set_pose_target(pose)
        # set velocity
        arm_group.set_max_velocity_scaling_factor(velocity_scale)
        # Plan and execute
        rospy.loginfo("Planning and going to the Cartesian Pose")
        return arm_group.go(wait=True)

    def reach_joint_angles(self, joint_angles, tolerance=0.01):
        arm_group = self.arm_group
        # Set the goal joint tolerance
        arm_group.set_goal_joint_tolerance(tolerance)
        arm_group.set_joint_value_target(joint_angles)
        return arm_group.go(wait=True)

    def reach_gripper_position(self, relative_position):
        # gripper_group = self.gripper_group
        # We only have to move this joint because all others are mimic!
        gripper_joint = self.robot.get_joint(self.gripper_joint_name)
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        try:
            val = gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos)
                                     + gripper_min_absolute_pos, True)
            return val
        except:
            return False

    def plan_trajectory_from_waypoints(self, waypoints, display_trajectory=False):
        # We want the Cartesian path to be interpolated at a resolution of 5 cm
        # which is why we will specify 0.05 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0
        (traj_plan, traj_fraction) = self.arm_group.compute_cartesian_path(waypoints, 0.05, 0.0)
        if display_trajectory:
            traj_visualize = moveit_msgs.msg.DisplayTrajectory()
            traj_visualize.trajectory_start = self.robot.get_current_state()
            traj_visualize.trajectory.append(traj_plan)
            self.display_trajectory_publisher.publish(traj_visualize)
        return traj_plan, traj_fraction

    def execute_trajectory_plan(self, trajectory_msg, velocity_scale=0.5, wait=True):
        self.arm_group.set_max_velocity_scaling_factor(velocity_scale)
        self.arm_group.execute(trajectory_msg, wait=wait)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

    def plan_to_cartesian_pose(self, pose, velocity_scale):
        self.arm_group.set_goal_position_tolerance(0.01)
        self.arm_group.set_goal_orientation_tolerance(0.1)  # 0.1 rad => 5.73 degrees
        self.arm_group.set_max_velocity_scaling_factor(velocity_scale)
        self.arm_group.set_pose_target(pose)
        plan = self.arm_group.plan()
        # print(plan)
        # print(plan.joint_trajectory)
        if not plan.joint_trajectory.points:
            return 0, []
        else:
            return 1, plan

    def add_collision_box(self, box_name, box_pose, box_size):
        if box_name == "base_table":
            box_pose = geometry_msgs.msg.PoseStamped()
            box_pose.header.frame_id = "base_link"
            box_pose.pose.orientation.w = 1.0
            box_pose.pose.position.x = 0.4
            box_pose.pose.position.y = 0.0
            box_pose.pose.position.z = -0.2
            box_size = (1, 2, 0.4)
        self.scene.add_box(box_name, box_pose, box_size)


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


def scanning_trajectory_waypoint_poses(scanning_waypoints):
    n = len(scanning_waypoints)
    trajectory_waypoints = np.empty((3 * n - 2, 3))
    for i in range(n - 1):
        step_i = (np.array(scanning_waypoints[i + 1]) - np.array(scanning_waypoints[i])) / 3
        for j in range(3):
            trajectory_waypoints[3 * i + j] = np.array(scanning_waypoints[i]) + j * step_i
    trajectory_waypoints[-1] = np.array(scanning_waypoints[-1])
    return trajectory_waypoints


def cartesian2spherical_coords(x_):
    r = (x_[0]**2 + x_[1]**2 + x_[2]**2)**0.5
    theta = np.arccos(x_[2]/r)
    sr = np.sin(theta) * r
    if sr != 0.0:
        phi = np.arctan2(x_[1]/sr, x_[0]/sr)
    else:
        phi = 0.0
    return [phi, theta, r]


def main():
    from semi_auto_grasping import generate_scanning_waypoints
    from pyquaternion import Quaternion
    move_my_gen3 = MoveMyGen3()
    success = move_my_gen3.is_init_success
    if success:
        rospy.loginfo('Moveit ROS node initialization complete. Move robot to safe start position.')
        success &= move_my_gen3.reach_named_position('home', velocity_scale=1)
        success &= move_my_gen3.reach_gripper_position(0)
        # ------------------
        poi_in_rob = Quaternion(0.0, 0.7, -0.023, 0.04)
        pre_d = 0.2
        upside_thetas = [0.0, 0.174, 0.348, 0.522]
        downside_thetas = [1.57, 1.39, 1.22]
        phis_1 = [1.74, 2.356]
        phis_2 = [4.53, 3.927]
        phis = [1.74, 2.356, np.pi, 4.53, 3.927]

        start_pose = view_param2cart_pose(poi_in_rob, [np.pi, np.pi / 4, pre_d])
        success &= move_my_gen3.reach_cartesian_pose(start_pose, velocity_scale=1)
        # move_my_gen3.arm_group.get_current_joint_values()

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
                        current_p_ = [current_p.x - poi_in_rob.x, current_p.y - poi_in_rob.y, current_p.z - poi_in_rob.z]
                        current_pose_param = cartesian2spherical_coords(current_p_)
                        print(current_pose_param)
                        trajectory_params = scanning_trajectory_waypoint_poses([current_pose_param, view_params])
                        print(trajectory_params)
                        reverse_trajectory_params = scanning_trajectory_waypoint_poses(
                            [view_params, [np.pi, np.pi / 4, pre_d]])
                        trajectory_poses = generate_scanning_waypoints(poi_in_rob, trajectory_params)
                        reverse_trajectory_poses = generate_scanning_waypoints(poi_in_rob, reverse_trajectory_params)
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
        rospy.spin()
    return success


def test():
    move_my_gen3 = MoveMyGen3()
    move_my_gen3.add_collision_box("base_table", [], [])
    rospy.spin()


if __name__ == "__main__":
    test()
    # task_success = main()
    # print('Task execution successful: ', task_success)
