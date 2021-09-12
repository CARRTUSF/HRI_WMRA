# /ar_marker_8 /camera_depth_optical_frame
# At time 1629230872.068
# - Translation: [-0.426, 0.055, 0.410]
# - Rotation: in Quaternion [-0.567, 0.710, -0.302, 0.288]
#             in RPY (radian) [-2.283, 0.066, -1.763]
#             in RPY (degree) [-130.814, 3.789, -101.003]

# /base_link /ar_marker_8
# At time 1629224772.773
# - Translation: [0.605, 0.070, -0.022]
# - Rotation: in Quaternion [0.011, 0.017, -0.708, 0.706]
#             in RPY (radian) [-0.008, 0.039, -1.574]
#             in RPY (degree) [-0.482, 2.221, -90.169]


from utils import pose_propagation_7d
pose_cam_in_mkr = [-0.334, -0.853, 0.492, 0.841, -0.172, 0.131, -0.496]
pose_mkr_in_rob = [0.592, 0.102, -0.028, 0.010, 0.016, -0.686, 0.727]
pose_cam_in_rob = pose_propagation_7d(pose_cam_in_mkr, pose_mkr_in_rob)
print('pose: x, y, z, wx, wy, wz, w')
print(pose_cam_in_rob)

# from pyquaternion import Quaternion
# cam_in_mkr_p_q = Quaternion(0.0, -0.426, 0.055, 0.410)
# cam_in_mkr_q = Quaternion(0.288, -0.567, 0.710, -0.302)
#
# mkr_in_rob_p_q = Quaternion(0.0, 0.605, 0.070, -0.022)
# mkr_in_rob_q = Quaternion(0.706, 0.011, 0.017, -0.708)
#
# cam_in_rob_q = mkr_in_rob_q * cam_in_mkr_q
#
# print 'camera in robot frame rotation'
# print cam_in_rob_q
#
# mkr_in_rob_q_ = mkr_in_rob_q.conjugate
# print 'robot in marker frame rotation'
# print mkr_in_rob_q_
#
# cam_in_mkr_in_rob_p_q = mkr_in_rob_q * cam_in_mkr_p_q * mkr_in_rob_q_
# cam_in_rob_p_q = mkr_in_rob_p_q + cam_in_mkr_in_rob_p_q
# print 'camera in robot translation'
# print cam_in_rob_p_q
# print 'camera in robot pose'
# print (cam_in_rob_p_q.x, cam_in_rob_p_q.y, cam_in_rob_p_q.z, cam_in_rob_q)
# print '----------------------------'
# rot_mkr_in_rob = quaternion.as_rotation_matrix(mkr_in_rob_q)
# print rot_mkr_in_rob
# p_cam_in_mkr = np.array([0.246, -0.260, 0.764]).reshape((3, 1))
# print p_cam_in_mkr
# p_cam_in_rob = np.matmul(rot_mkr_in_rob, p_cam_in_mkr)
# print p_cam_in_rob
# rq = Quaternion(0.7071068, 0, 0.7071068, 0)
# p = Quaternion(0.0, 0.0, 0.0, 1.0)
# rq_ = Quaternion(0.7071068, 0, -0.7071068, 0)
# p_ = rq*p*rq_
# print p_
# 0.227181572, 0.772881662, 0.5063396280000001, quaternion(0.225718, -0.251663, 0.777028, -0.53125)

# static_transform_publisher x y z qx qy qz qw frame_id child_frame_id  period_in_ms
# rosrun tf static_transform_publisher 0.227181572 0.772881662 0.506339628 -0.251663 0.777028 -0.53125 0.225718 /base_link /camera_depth_optical_frame 50

