# /ar_marker_8 /camera_depth_optical_frame
# At time 1628099649.316
# - Translation: [0.112, 0.663, 0.517]
# - Rotation: in Quaternion [0.003, 0.907, -0.420, 0.033]
#             in RPY (radian) [-2.274, 0.062, -3.119]
#             in RPY (degree) [-130.310, 3.525, -178.726]


# /base_link /ar_marker_8
# At time 1628098565.224
# - Translation: [0.425, 0.194, -0.027]
# - Rotation: in Quaternion [0.009, -0.058, 0.960, -0.274]
#             in RPY (radian) [-0.117, 0.015, -2.587]
#             in RPY (degree) [-6.700, 0.871, -148.245]

from pyquaternion import Quaternion


cam_in_mkr_p_q = Quaternion(0.0, 0.112, 0.663, 0.517)
cam_in_mkr_q = Quaternion(0.033, 0.003, 0.907, -0.420)

mkr_in_rob_p_q = Quaternion(0.0, 0.425, 0.194, -0.027)
mkr_in_rob_q = Quaternion(-0.274, 0.009, -0.058, 0.960)

cam_in_rob_q = mkr_in_rob_q * cam_in_mkr_q

print 'camera in robot frame rotation'
print cam_in_rob_q

mkr_in_rob_q_ = mkr_in_rob_q.conjugate
print 'robot in marker frame rotation'
print mkr_in_rob_q_

cam_in_mkr_in_rob_p_q = mkr_in_rob_q * cam_in_mkr_p_q * mkr_in_rob_q_
cam_in_rob_p_q = mkr_in_rob_p_q + cam_in_mkr_in_rob_p_q
print 'camera in robot translation'
print cam_in_rob_p_q
print 'camera in robot pose'
print (cam_in_rob_p_q.x, cam_in_rob_p_q.y, cam_in_rob_p_q.z, cam_in_rob_q)
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
#0.227181572, 0.772881662, 0.5063396280000001, quaternion(0.225718, -0.251663, 0.777028, -0.53125)

# static_transform_publisher x y z qx qy qz qw frame_id child_frame_id  period_in_ms
# rosrun tf static_transform_publisher 0.227181572 0.772881662 0.506339628 -0.251663 0.777028 -0.53125 0.225718 /base_link /camera_depth_optical_frame 50

