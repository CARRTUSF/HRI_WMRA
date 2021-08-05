import os
import sys
import geometry_msgs.msg
import rospy
import numpy as np
import moveit_commander
import moveit_msgs.msg

os.environ["ROS_NAMESPACE"] = "/my_gen3/"


class MoveMyGen3(object):

    def __init__(self):
        # Initialize the node
        super(MoveMyGen3, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        # rospy.init_node('kinova_gen3_moveit')
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

    def execute_trajectory_plan(self, trajectory_msg, wait=True):
        self.arm_group.execute(trajectory_msg, wait=wait)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

    def plan_to_cartesian_pose(self, pose, tolerance, velocity_scale):
        self.arm_group.set_goal_position_tolerance(tolerance[0])
        self.arm_group.set_goal_orientation_tolerance(tolerance[1])  # 0.1 rad => 5.73 degrees
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


def jacobian_spherical2cartesian(_cos_a, _sin_a, spherical_coordinates_, angle_representation='radians'):
    phi = spherical_coordinates_[0]
    theta = spherical_coordinates_[1]
    r = spherical_coordinates_[2]
    if angle_representation == 'radians':
        if theta == 0.0:
            phi = np.pi
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
    else:
        if theta == 0.0:
            phi = 180
        cos_p = np.cos(np.deg2rad(phi))
        sin_p = np.sin(np.deg2rad(phi))
        cos_t = np.cos(np.deg2rad(theta))
        sin_t = np.sin(np.deg2rad(theta))
    jacobian__ = np.zeros((6, 3))
    jacobian__[0, 0] = - r * sin_t * (_cos_a * sin_p + _sin_a * cos_p)
    jacobian__[0, 1] = r * cos_t * (_sin_a * sin_p - _cos_a * cos_p)
    jacobian__[0, 2] = sin_t * (_cos_a * cos_p - _sin_a * sin_p)
    jacobian__[1, 0] = r * sin_t * (_cos_a * cos_p - _sin_a * sin_p)
    jacobian__[1, 1] = - r * cos_t * (_cos_a * sin_p + _sin_a * cos_p)
    jacobian__[1, 2] = sin_t * (_cos_a * sin_p + _sin_a * cos_p)
    jacobian__[2, 1] = r * sin_t
    jacobian__[2, 2] = cos_t
    jacobian__[3, 1] = _cos_a * sin_p + _sin_a * cos_p
    jacobian__[4, 1] = _sin_a * sin_p - _cos_a * cos_p
    jacobian__[5, 0] = 1
    return jacobian__
