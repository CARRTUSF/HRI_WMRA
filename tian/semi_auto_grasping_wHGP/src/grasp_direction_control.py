import rospy
import numpy as np
import pygame
import sys
from kortex_driver.msg import *
from std_msgs.msg import Empty as std_Empty
from sensor_msgs.msg import Image
from tt_move_gen3 import MoveMyGen3
from semi_auto_grasping import trajectory_params2poses
from pyquaternion import Quaternion
from cv_bridge import CvBridge
import cv2


bridge = CvBridge()
color_img = np.zeros((100, 100))


def color_callback(color_msg):
    global color_img
    color_img = bridge.imgmsg_to_cv2(color_msg, 'rgb8')


def cartesian2spherical_coords(x_):
    r = (x_[0]**2 + x_[1]**2 + x_[2]**2)**0.5
    theta = np.arccos(x_[2]/r)
    sr = np.sin(theta) * r
    if sr != 0.0:
        phi = np.arctan2(x_[1]/sr, x_[0]/sr)
    else:
        phi = 0.0
    return [phi, theta, r]


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


if __name__ == '__main__':
    # rospy.init_node('grasp_direction_control', anonymous=True)
    move_my_gen3 = MoveMyGen3()
    success = move_my_gen3.is_init_success
    # success &= move_my_gen3.reach_named_position('home')
    success &= move_my_gen3.reach_gripper_position(0)
    x0 = 0.7
    y0 = -0.053
    z0 = 0.049
    n_xy0 = (x0**2 + y0**2)**0.5
    pre_d = 0.25
    c0 = x0 / n_xy0
    s0 = y0 / n_xy0
    t_b0 = np.array([[c0, -s0, 0, x0],
                     [s0, c0, 0, y0],
                     [0, 0, 1, z0],
                     [0, 0, 0, 1]])
    t_0b = np.array([[c0, s0, 0, -(x0*c0+y0*s0)],
                     [-s0, c0, 0, x0*s0-y0*c0],
                     [0, 0, 1, -z0],
                     [0, 0, 0, 1]])
    r_0b = np.array([[c0, -s0, 0],
                     [s0, c0, 0],
                     [0, 0, 1]])
    pre_grasp_param = np.array([[np.pi, np.pi/4, pre_d]])
    pre_grasp_pose = trajectory_params2poses(Quaternion(0, x0, y0, z0), pre_grasp_param)[0]
    print(pre_grasp_pose)
    # success &= move_my_gen3.reach_cartesian_pose(pre_grasp_pose)
    current_p = move_my_gen3.arm_group.get_current_pose().pose.position
    print(current_p)
    print('&&&&&&&&&&&&&&&&&')
    current_p_b = np.array([[current_p.x], [current_p.y], [current_p.z], [1]])
    current_p_0 = np.matmul(t_0b, current_p_b).flatten()
    print(current_p_0)
    current_spherical_coords = cartesian2spherical_coords(current_p_0)
    print(current_spherical_coords)
    print(pre_grasp_param)
    print('-----------------------------------------')
    if success:
        twist_command_pub = rospy.Publisher('/my_gen3/in/cartesian_velocity', kortex_driver.msg.TwistCommand, queue_size=5)
        stop_command_pub = rospy.Publisher('/my_gen3/in/stop', std_Empty, queue_size=1)
        rospy.Subscriber('/camera/color/image_raw', Image, color_callback, queue_size=1)
        twist_command = kortex_driver.msg.TwistCommand()
        twist_command.reference_frame = 3
        stop_command = std_Empty()
        print('stop command: ', stop_command)
        # ----------------- user interface ---------------------
        pygame.init()
        # fps = 40
        # fpsclock = pygame.time.Clock()
        screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Grasp Direction Control Input")
        arrow_img = pygame.image.load('arrow.png')
        White = (255, 255, 255)
        rate = rospy.Rate(40)
        while not rospy.is_shutdown():
            # ------------------ default velocity ----------------------
            phi_dot = 0.0
            theta_dot = 0.0
            stop = True
            # ------------------ user interface --------------------------
            # ------------ hand camera view
            screen.fill(White)
            # print('color image size: ', color_img.shape)
            # print(color_img)
            color_display_img = pygame.surfarray.make_surface(color_img)
            color_display_img = pygame.transform.rotate(color_display_img, 90)
            color_display_img = pygame.transform.flip(color_display_img, 0, 1)
            screen.blit(color_display_img, (0, 0))

            for eve in pygame.event.get():
                if eve.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            up_arrow_img = arrow_img.copy()
            key_input = pygame.key.get_pressed()
            unit_angular_velocity = 4*np.pi/180
            if key_input[pygame.K_LEFT]:
                phi_dot = -unit_angular_velocity
                left_arrow_img = pygame.transform.rotate(up_arrow_img, 90)
                screen.blit(left_arrow_img, (0, 190))
                stop = False
            if key_input[pygame.K_UP]:
                theta_dot = unit_angular_velocity
                screen.blit(up_arrow_img, (270, 0))
                stop = False
            if key_input[pygame.K_RIGHT]:
                phi_dot = unit_angular_velocity
                right_arrow_img = pygame.transform.rotate(up_arrow_img, -90)
                screen.blit(right_arrow_img, (520, 190))
                stop = False
            if key_input[pygame.K_DOWN]:
                theta_dot = -unit_angular_velocity
                down_arrow_img = pygame.transform.rotate(up_arrow_img, 180)
                screen.blit(down_arrow_img, (270, 360))
                stop = False
            if key_input[pygame.K_ESCAPE]:
                pygame.quit()
                sys.exit()
            pygame.display.update()
            # --------------------------------------------------------------
            # ---------------------- calculate velocity command -----------------------
            if stop:
                stop_command_pub.publish(stop_command)
            else:
                current_p = move_my_gen3.arm_group.get_current_pose().pose.position
                current_p_b = np.array([[current_p.x], [current_p.y], [current_p.z], [1]])
                current_p_0 = np.matmul(t_0b, current_p_b).flatten()
                current_spherical_coords = cartesian2spherical_coords(current_p_0)
                print(current_spherical_coords)
                j__63 = jacobian_spherical2cartesian(c0, s0, current_spherical_coords, angle_representation='radians')
                twist__61 = np.matmul(j__63, np.array([[phi_dot], [theta_dot], [0.0]]))
                # print('velocity command')
                # fpsclock.tick(fps)
                twist_command.twist.linear_x = twist__61[0, 0]
                twist_command.twist.linear_y = twist__61[1, 0]
                twist_command.twist.linear_z = twist__61[2, 0]
                twist_command.twist.angular_x = twist__61[3, 0]
                twist_command.twist.angular_y = twist__61[4, 0]
                twist_command.twist.angular_z = twist__61[5, 0]
                # print(twist_command)
                twist_command_pub.publish(twist_command)
            rate.sleep()
