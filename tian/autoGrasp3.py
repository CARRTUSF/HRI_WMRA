#! /usr/bin/env python

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import numpy
import baxter_interface
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from baxter_interface import CHECK_VERSION
from helper_func import pose2transformation, transformation2pose
from KinectCalibration import kinect2baxter_transform
from pyquaternion import Quaternion

PoseArray = geometry_msgs.msg.PoseArray()
object_index = -1
grasp_index = -1
mode = 1
attempt = -1
feedback = geometry_msgs.msg.Vector3()
cloud_points=[]
pixel_x = -1
pixel_y = -1


def get_object_poses(ope_inCam):
    global PoseArray
    if len(ope_inCam.poses) != 0:
        for i in range(0, len(ope_inCam.poses)):
            while not len(PoseArray.poses) == len(ope_inCam.poses):
                if len(PoseArray.poses) > len(ope_inCam.poses):
                    PoseArray.poses.pop(-1)
                else:
                    PoseArray.poses.append(geometry_msgs.msg.Pose())

            PoseArray.poses[i] = ope_inCam.poses[i]
    else:
        PoseArray = geometry_msgs.msg.PoseArray()


def frame_trans(pose_kinect):
    kinbax_trans = kinect2baxter_transform()
    T_OinK = pose2transformation(pose_kinect)
    T_OinB = numpy.matmul(kinbax_trans, T_OinK)
    object_pose = transformation2pose(T_OinB)
    return object_pose


def get_user_input(user_input):
    global object_index
    global grasp_index
    global mode
    global attempt
    global pixel_x
    global pixel_y
    mode = user_input.z
    attempt = user_input.w
    if mode == 4:
        if user_input.x != -1 and user_input.y != -1:
            pixel_x = int(user_input.x * 640)
            pixel_y = int(user_input.y * 480)
    else:
        object_index = user_input.x
        grasp_index = user_input.y


def get_pointcloud(msg2):
    global cloud_points
    if len(msg2.data) != 0:
        cloud_points = list(pc2.read_points(msg2, field_names = ("x", "y", "z")))
    else:
        print("no point cloud data")


def find_selected_point(pixelx, pixely, pointcloud):
    point = geometry_msgs.msg.Vector3()
    if len(pointcloud) != 0:
        selected_point_index = pixelx + pixely * 640
        point.x = pointcloud[selected_point_index][0]
        point.y = pointcloud[selected_point_index][1]
        point.z = pointcloud[selected_point_index][2]
        null_rotation = Quaternion()
        pose_inCam = [point.x, point.y, point.z, null_rotation]
        print(pose_inCam)
        pose_inRobot = frame_trans(pose_inCam)
        print(pose_inRobot)
        point.x = pose_inRobot[0]
        point.y = pose_inRobot[1]
        point.z = pose_inRobot[2]
    else:
        point = (-1, -1, -1)
    return point


def main():
    global PoseArray
    global object_index
    global grasp_index
    global mode
    global attempt
    global feedback
    global pixel_x
    global pixel_y
    global cloud_points
    feedback.x = -1
    feedback.y = -1
    feedback.z = 0
    right_start = {
        'right_e0': 1.3430001797936797,
        'right_e1': 2.244980883070303,
        'right_s0': -0.9499176028980425,
        'right_s1': -0.6661311571392409,
        'right_w0': -1.0684176187621908,
        'right_w1': 0.8966117705190243,
        'right_w2': -0.09203884727312482
    }
    # print('Getting robot state...')
    # First initialize moveit_commander and rospy.
    print "============ Starting autonomous grasping node"
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_autonomous_grasp', anonymous=True)
    rospy.Subscriber('/ope_inCamFrame', geometry_msgs.msg.PoseArray, get_object_poses)
    rospy.Subscriber('/user_selected_grasp', geometry_msgs.msg.Quaternion, get_user_input)
    rospy.Subscriber('/camera/depth_registered/points',PointCloud2, get_pointcloud)
    feedback_pub = rospy.Publisher('auto_grasp_feedback', geometry_msgs.msg.Vector3, queue_size=10)

    rate = rospy.Rate(20)
    previous_attempt = -2

    # Instantiate a RobotCommander object. This object is an interface to the robot as a whole.
    robot = moveit_commander.RobotCommander()

    # Instantiate a PlanningSceneInterface object. This object is an interface to the world surrounding the robot.
    # scene = moveit_commander.PlanningSceneInterface()

    # Instantiate a MoveGroupCommander object. This object is an interface to one group of joints.
    # In this case the group is the joints in the right arm. This interface can be used to plan and execute motions on the right arm.
    # the group_name can be found http://sdk.rethinkrobotics.com/wiki/MoveIt_Tutorial#ikfast_IKFast
    group = moveit_commander.MoveGroupCommander("right_arm")

    # create this DisplayTrajectory publisher which is used below to publish trajectories for RVIZ to visualize.
    # display_trajectory_publisher = rospy.Publisher(
    #                                '/move_group/display_planned_path',
    #                                moveit_msgs.msg.DisplayTrajectory,queue_size=10)

    grip_right = baxter_interface.Gripper('right', CHECK_VERSION)
    grip_right.calibrate()
    arm_right = baxter_interface.Limb('right')
    # print "============ Waiting for RVIZ..."
    # rospy.sleep(10)
    print "============ Starting program... "

    #####getting basic information#####
    print "============ Reference frame: %s" % group.get_planning_frame()  # world frame
    print "============ End_effector frame: %s" % group.get_end_effector_link()  # end_effector
    print "============ Robot Groups:"  # get a list of all groups in the robot
    print robot.get_group_names()
    # print "============ Printing robot state"  # robot state
    # print robot.get_current_state()
    print "============"
    # ---------------------------------------------------
    # set the 6D goal pose here
    # print object_pose
    print "Initialization complete, waiting for command"
    while not rospy.is_shutdown():
        if mode == 1:
            if object_index != -1 and attempt != previous_attempt and len(PoseArray.poses) != 0:
                print("start grasp")
                previous_attempt = attempt
                print(object_index)
                print(attempt)
                pose_target = geometry_msgs.msg.Pose()
                ori_quat = Quaternion()
                pos_x = PoseArray.poses[int(object_index)].position.x
                pos_y = PoseArray.poses[int(object_index)].position.y
                pos_z = PoseArray.poses[int(object_index)].position.z
                ori_quat[1] = PoseArray.poses[int(object_index)].orientation.x
                ori_quat[2] = PoseArray.poses[int(object_index)].orientation.y
                ori_quat[3] = PoseArray.poses[int(object_index)].orientation.z
                ori_quat[0] = PoseArray.poses[int(object_index)].orientation.w
                selected_object_pose = [pos_x, pos_y, pos_z, ori_quat]
                object_pose = frame_trans(selected_object_pose)
                print(selected_object_pose)
                # ----------------------++++++++++----------------------------#
                if grasp_index == 0:
                    # pose_target.position.x = object_pose[0] - 0.05
                    # pose_target.position.y = object_pose[1]
                    # pose_target.position.z = object_pose[2]
                    pose_target.orientation.x = 0.0  # object_pose[3][1]
                    pose_target.orientation.y = 0.7071067811865475  # object_pose[3][2]
                    pose_target.orientation.z = 0.0  # object_pose[3][3]
                    pose_target.orientation.w = 0.7071067811865475  # object_pose[3][0]
                elif grasp_index == 1:
                    # pose_target.position.x = object_pose[0] - 0.04
                    # pose_target.position.y = object_pose[1] - 0.04
                    # pose_target.position.z = object_pose[2]
                    pose_target.orientation.x = -0.270598  # 0.6532815#grasp_orientation.normalized[1]#object_pose[3][1]
                    pose_target.orientation.y = 0.6532815  # 0.270598#grasp_orientation.normalised[2]#object_pose[3][2]
                    pose_target.orientation.z = 0.270598  # 0.6532815#grasp_orientation.normalised[3]#object_pose[3][3]
                    pose_target.orientation.w = 0.6532815  # -0.270598#grasp_orientation.normalised[0]#object_pose[3][0]
                elif grasp_index == 2:
                    # pose_target.position.x = object_pose[0]
                    # pose_target.position.y = object_pose[1] - 0.05
                    # pose_target.position.z = object_pose[2]
                    pose_target.orientation.x = 0.0 # object_pose[3][1]
                    pose_target.orientation.y = 1.0  # object_pose[3][2]
                    pose_target.orientation.z = 0.0  # object_pose[3][3]
                    pose_target.orientation.w = 0.0  # object_pose[3][0]
                    #pose_target.orientation.x = -0.5  # object_pose[3][1]
                    #pose_target.orientation.y = 0.5  # object_pose[3][2]
                    #pose_target.orientation.z = 0.5  # object_pose[3][3]
                    #pose_target.orientation.w = 0.5  # object_pose[3][0]
                else:
                    pass
                # print pose_target
                # print grasp_index
                # print "execution successed"
                k = 0.05
                h = 0.05
                gripper_oriquat = Quaternion()
                gripper_oriquat[1] = pose_target.orientation.x
                gripper_oriquat[2] = pose_target.orientation.y
                gripper_oriquat[3] = pose_target.orientation.z
                gripper_oriquat[0] = pose_target.orientation.w
                gripper_orirot = gripper_oriquat.normalised.rotation_matrix
                pose_target.position.x = object_pose[0] + gripper_orirot[0, 2] * k + 0.02
                pose_target.position.y = object_pose[1] + gripper_orirot[1, 2] * k
                pose_target.position.z = object_pose[2] + gripper_orirot[2, 2] * k + 0.03
                print(pose_target)
                group.set_pose_target(pose_target)
                found_plan = group.plan()
                print found_plan
                if found_plan.joint_trajectory.points == []:
                    feedback.x = 0
                    feedback.z += 1
                    feedback_pub.publish(feedback)
                    rospy.sleep(0.5)
                else:
                    pose_target.position.x = object_pose[0] - gripper_orirot[0, 2] * h + 0.02
                    pose_target.position.y = object_pose[1] - gripper_orirot[1, 2] * h
                    pose_target.position.z = object_pose[2] - gripper_orirot[2, 2] * h + 0.03
                    group.set_pose_target(pose_target)
                    group.plan()
                    print("approaching to pre-grasp")
                    feedback.x = 1
                    feedback.y = 0
                    feedback.z += 1
                    feedback_pub.publish(feedback)
                    grip_right.open()
                    group.set_max_velocity_scaling_factor(0.6)
                    group.go(wait=True)#***************************************************************************
                    #group.stop()
                    group.clear_pose_targets()
                    rospy.sleep(0.5)
                    print("grasping")
                    grasp_target = geometry_msgs.msg.Pose()
                    grasp_target.orientation = pose_target.orientation
                    grasp_target.position.x = object_pose[0] + gripper_orirot[0, 2] * k + 0.02
                    grasp_target.position.y = object_pose[1] + gripper_orirot[1, 2] * k
                    grasp_target.position.z = object_pose[2] + gripper_orirot[2, 2] * k + 0.03
                    group.set_pose_target(grasp_target)
                    found_plan2 = group.plan()
                    if found_plan2.joint_trajectory.points == []:
                        print("can't reach out")
                    else:
                        group.set_max_velocity_scaling_factor(0.4)
                        group.go(wait=True)#***********************************************************************
                        grip_right.close()
                        #group.stop()
                        group.clear_pose_targets()
                        print("done")
                        feedback.y = 1
                        feedback.z += 1
                        feedback_pub.publish(feedback)
                        # Execute_grasp = 0
                        # reset = 0
                        rospy.sleep(0.5)
            else:
                tempx = feedback.x
                if tempx != -1:
                    feedback.x = -1
                    feedback.y = -1
                    feedback.z += 1
                    feedback_pub.publish(feedback)
                    rospy.sleep(0.5)
        elif mode == 3:
            if attempt != previous_attempt:
                previous_attempt = attempt
                feedback.x = 2
                feedback.y = 0
                feedback.z += 1
                feedback_pub.publish(feedback)
                print("move to nutrual")
                grip_right.open()
                #arm_right.move_to_joint_positions(right_start)
                group.set_joint_value_target({'right_e0': 1.3430001797936797, 'right_e1': 2.244980883070303, 'right_s0': -0.9499176028980425, 'right_s1': -0.6661311571392409, 'right_w0': -1.0684176187621908, 'right_w1': 0.8966117705190243,'right_w2': -0.09203884727312482})
                group.plan()
                group.set_max_velocity_scaling_factor(0.7)
                group.go(wait=True)#*********************************************************************************
                #group.stop()
                group.clear_pose_targets()
                feedback.y = 1
                feedback.z += 1
                feedback_pub.publish(feedback)
                rospy.sleep(0.5)
        elif mode == 4:
            if attempt != previous_attempt and pixel_x != -1 and pixel_y != -1:
                print('got placing cmd')
                previous_attempt = attempt
                print(len(cloud_points))
                print('-----------------------------------')
                selected_position = find_selected_point(pixel_x,pixel_y,cloud_points)
                print(selected_position)
                if selected_position != (-1, -1, -1):
                    group.clear_pose_target(group.get_end_effector_link())
                    #pose_to_move2 = group.get_current_pose(group.get_end_effector_link()).pose
                    pose_to_move2_dict = arm_right.endpoint_pose()
                    print(pose_to_move2_dict)
                    pose_to_move2 = geometry_msgs.msg.Pose()
                    pose_to_move2.position.x = selected_position.x
                    pose_to_move2.position.y = selected_position.y
                    pose_to_move2.position.z = pose_to_move2_dict['position'].z
                    pose_to_move2.orientation.x = pose_to_move2_dict['orientation'].x
                    pose_to_move2.orientation.y = pose_to_move2_dict['orientation'].y
                    pose_to_move2.orientation.z = pose_to_move2_dict['orientation'].z
                    pose_to_move2.orientation.w = pose_to_move2_dict['orientation'].w
                    print(pose_to_move2)
                    group.set_pose_target(pose_to_move2)
                    found_plan3 = group.plan()
                    if found_plan3.joint_trajectory.points ==[]:
                        print("cannot move there")
                    else:
                        group.clear_pose_target(group.get_end_effector_link())
                        pose_to_move2.position.z += 0.1
                        group.set_pose_target(pose_to_move2)
                        group.plan()
                        print("going to the selected palce")
                        group.set_max_velocity_scaling_factor(0.5)
                        group.go(wait=True)#***********************************************************************************
                        rospy.sleep(1)
                        pose_to_place = pose_to_move2
                        pose_to_place.position.z += -0.1
                        group.clear_pose_target(group.get_end_effector_link())
                        group.set_pose_target(pose_to_place)
                        #group.set_max_velocity_scaling_factor(0.3)
                        group.plan()
                        print("2nd stage")
                        #group.go()#***********************************************************************************

        else:
            if attempt != previous_attempt:
                previous_attempt = attempt
                feedback.x = 3
                feedback.y = 0
                feedback.z += 1
                feedback_pub.publish(feedback)
                rospy.sleep(0.5)
        rate.sleep()
    moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    main()
