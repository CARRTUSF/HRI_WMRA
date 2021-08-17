import numpy as np
from pyquaternion import Quaternion


def trans_matrix_from_7d_pose(x, y, z, wx, wy, wz, w):
    q_ = Quaternion(w, wx, wy, wz)
    t__ = np.zeros((4, 4), dtype=np.float64)
    t__[0, 3] = x
    t__[1, 3] = y
    t__[2, 3] = z
    t__[3, 3] = 1
    r__ = q_.rotation_matrix
    t__[:3, :3] = r__
    return t__


def cartesian2spherical_coords(x_):
    r = (x_[0]**2 + x_[1]**2 + x_[2]**2)**0.5
    theta = np.arccos(x_[2]/r)
    sr = np.sin(theta) * r
    if sr != 0.0:
        phi = np.arctan2(x_[1]/sr, x_[0]/sr)
    else:
        phi = 0.0
    return [phi, theta, r]


def jaco_zyx_angles_from_rotation_matrix(r__):
    # base on ZYX euler angles: R_zyx(theta_x, theta_y, theta_z) = Rz(theta_z)Ry(theta_y)Rx(theta_x)
    if abs(r__[0, 0]) > 1e-10 or abs(r__[1, 0]) > 1e-10:
        c2 = (r__[0, 0] ** 2 + r__[1, 0] ** 2) ** 0.5
        theta_y = np.arctan2(-r__[2, 0], c2) * 180/np.pi
        theta_z = np.arctan2(r__[1, 0]/c2, r__[0, 0]/c2) * 180/np.pi
        theta_x = np.arctan2(r__[2, 1]/c2, r__[2, 2]/c2) * 180/np.pi
    else:
        theta_y = 90.0
        theta_x = 0.0
        theta_z = np.arctan2(r__[0, 1], r__[1, 1])
    return [theta_x, theta_y, theta_z]


def fundamental_axis_rotation_matrix(_theta, _axis=1):
    c = np.cos(np.deg2rad(_theta))
    s = np.sin(np.deg2rad(_theta))
    if _axis == 1:
        r__ = np.matrix([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif _axis == 2:
        r__ = np.matrix([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    else:
        r__ = np.matrix([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    return r__


def jacobian_spherical2cartesian(_alpha_, _phi_, _theta_, _r, angle_representation='radians'):
    if angle_representation == 'radians':
        if _theta_ == 0.0:
            _phi_ = np.pi
        cos_a = np.cos(_alpha_)
        sin_a = np.sin(_alpha_)
        cos_p = np.cos(_phi_)
        sin_p = np.sin(_phi_)
        cos_t = np.cos(_theta_)
        sin_t = np.sin(_theta_)
    else:
        if _theta_ == 0.0:
            _phi_ = 180
        cos_a = np.cos(np.deg2rad(_alpha_))
        sin_a = np.sin(np.deg2rad(_alpha_))
        cos_p = np.cos(np.deg2rad(_phi_))
        sin_p = np.sin(np.deg2rad(_phi_))
        cos_t = np.cos(np.deg2rad(_theta_))
        sin_t = np.sin(np.deg2rad(_theta_))
    jacobian__ = np.zeros((6, 3))
    jacobian__[0, 0] = - _r * sin_t * (cos_a * sin_p + sin_a * cos_p)
    jacobian__[0, 1] = _r * cos_t * (sin_a * sin_p - cos_a * cos_p)
    jacobian__[0, 2] = sin_t * (cos_a * cos_p - sin_a * sin_p)
    jacobian__[1, 0] = _r * sin_t * (cos_a * cos_p - sin_a * sin_p)
    jacobian__[1, 1] = - _r * cos_t * (cos_a * sin_p + sin_a * cos_p)
    jacobian__[1, 2] = sin_t * (cos_a * sin_p + sin_a * cos_p)
    jacobian__[2, 1] = _r * sin_t
    jacobian__[2, 2] = cos_t
    jacobian__[3, 1] = cos_a * sin_p + sin_a * cos_p
    jacobian__[4, 1] = sin_a * sin_p - cos_a * cos_p
    jacobian__[5, 0] = 1
    return jacobian__


if __name__ == '__main__':
    print('test trans_matrix_from_7d_pose(0.485, 0.228, 0.356, -0.369, 0.872, -0.280, 0.158)')
    t = trans_matrix_from_7d_pose(0.485, 0.228, 0.356, -0.369, 0.872, -0.280, 0.158)
    print(t)
    print('test ')
    q = Quaternion(0.152956, 0.654488, 0.498251, 0.547719)
    q = q.normalised
    print(q)
    print(q.w)
    rotation_matrix = q.rotation_matrix
    print(rotation_matrix)
    r__x = fundamental_axis_rotation_matrix(116.353, _axis=1)
    r__y = fundamental_axis_rotation_matrix(-33.121, _axis=2)
    r__z = fundamental_axis_rotation_matrix(96.56, _axis=3)
    r_zyx = np.matmul(np.matmul(r__z, r__y), r__x)
    print(r_zyx)
    # [-0.0965001, 0.4846447, 0.8693718;rotation_matrix
    # 0.8197522, -0.4567010, 0.3455873;
    # 0.5645300, 0.7460186, -0.3532169]
    zyx_angles = jaco_zyx_angles_from_rotation_matrix(rotation_matrix)
    print(zyx_angles)
    print(jacobian_spherical2cartesian(0, -45, 45, 0.2, angle_representation='degrees'))

# - Translation: [0.485, 0.228, 0.356]
# - Rotation: in Quaternion [-0.369, 0.872, -0.280, 0.158]
# q1 = np.quaternion(0.043, 0.709, -0.703, -0.025)
#
# print(q1.w)
# r1 = quaternion.as_rotation_matrix(q1)
# T1 = np.identity(4)
#
# T1[:3, :3] = r1
# T1[0, 3] = 0.669
# T1[1, 3] = -0.424
# T1[2, 3] = 0.271
# T1[3, 3] = 1.0
# print(T1)
#
# q2 = np.quaternion(0.203, -0.582, 0.769, 0.171)
#
# r2 = quaternion.as_rotation_matrix(q2)
# T2 = np.identity(4)
#
# T2[:3, :3] = r2
# T2[0, 3] = 0.603
# T2[1, 3] = -0.682
# T2[2, 3] = 0.211
# T2[3, 3] = 1.0
# print(T2)
#
# T21 = np.matmul(np.linalg.inv(T1), T2)
# print(T21)
#
#
# q4 = np.quaternion(0.385, -0.649, 0.651, -0.084)
#
# r4 = quaternion.as_rotation_matrix(q4)
# T4 = np.identity(4)
#
# T4[:3, :3] = r4
# T4[0, 3] = 0.319
# T4[1, 3] = -0.566
# T4[2, 3] = 0.175
# T4[3, 3] = 1.0
# print('t4')
# print(T4)
# print('T41')
# T41 = np.matmul(np.linalg.inv(T1), T4)
# print(T41)
