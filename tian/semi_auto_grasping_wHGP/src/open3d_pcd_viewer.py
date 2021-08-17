import open3d as o3d
import numpy as np


pcd = o3d.io.read_point_cloud('s3.pcd')
# print(pcd)
# # print(np.asarray(pcd.points))
# ds = np.linspace(-0.5, 0.5, 11)
# mx, my = np.meshgrid(ds, ds)
# print(mx)
# print(my)
# xyz = np.zeros((np.size(mx), 3))
# dxyz = np.ones(np.size(mx))
# dd = 0.1
# xyz[:, 0] = np.reshape(mx, -1)
# xyz[:, 1] = np.reshape(my, -1)
# print(xyz)
# xyz_all = np.copy(xyz)
# for i in range(1, 4):
#     new_xyz = np.copy(xyz)
#     new_xyz[:, 2] = np.reshape(dxyz*dd*i, -1)
#     xyz_all = np.append(xyz_all, new_xyz, axis=0)
# print(xyz_all)
# npcd = o3d.geometry.PointCloud()
# npcd.points = o3d.utility.Vector3dVector(xyz_all)
# npcd.paint_uniform_color([1, 0, 0])
# pcd_combined = pcd + npcd
# o3d.visualization.draw_geometries([pcd_combined])
#
# color_img = o3d.io.read_image('0_color.png')
# depth_img = o3d.io.read_image('0_depth.png')
# rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, convert_rgb_to_intensity=False)
# print(rgbd_img)
# camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
#                                                       fx=653.68229, fy=651.855994,
#                                                       cx=311.753415, cy=232.400954)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, camera_intrinsics)
o3d.visualization.draw_geometries([pcd])
# [653.68229, 0.0, 311.753415, 0.0, 651.855994, 232.400954, 0.0, 0.0, 1.0]