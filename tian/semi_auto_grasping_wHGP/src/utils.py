import numpy as np
import geometry_msgs.msg
from pyquaternion import Quaternion


def cartesian2spherical_coords(x_):
    r = (x_[0]**2 + x_[1]**2 + x_[2]**2)**0.5
    theta = np.arccos(x_[2]/r)
    sr = np.sin(theta) * r
    if sr != 0.0:
        phi = np.arctan2(x_[1]/sr, x_[0]/sr)
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