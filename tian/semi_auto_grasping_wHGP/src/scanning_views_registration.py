from open3d import open3d as o3d
import numpy as np
from utils import trans_matrix_from_7d_pose
from tt_pointcloud_utils import tt_crop_point_cloud, tt_cloud_clustering, tt_cloud_plane_removal, \
    display_inlier_outlier, tt_get_cluster_bb_params


def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


def create_pointcloud_from_rgbd_images(color_image_file, depth_image_file, _camera_intrinsic):
    _color_img = o3d.io.read_image(color_image_file)
    _depth_img = o3d.io.read_image(depth_image_file)
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        _color_img, _depth_img, convert_rgb_to_intensity=False)
    _pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_img, _camera_intrinsic)
    return _pointcloud


if __name__ == '__main__':
    poses = np.loadtxt("s_poses.txt")
    n_pcds = poses.shape[0]
    T_0_base = trans_matrix_from_7d_pose(poses[0][0], poses[0][1], poses[0][2],
                                         poses[0][3], poses[0][4], poses[0][5], poses[0][6])
    # camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
    #                                                      fx=920.8743, fy=921.56549,
    #                                                      cx=644.76465, cy=343.153)
    # print(camera_intrinsic)
    multi_view_pcds = []
    voxel_size = 0.003
    for i in range(n_pcds):
        cloud_file = 's' + str(i + 1) + '.pcd'
        print(cloud_file)
        cloud = o3d.io.read_point_cloud(cloud_file)
        print(cloud)
        cloud = cloud.voxel_down_sample(
            voxel_size=voxel_size)
        print(cloud)
        pcd_bk_removed = tt_crop_point_cloud([-0.5, 0.5], [-0.5, 0.5], [0, 0.6], cloud)
        pcd_bk_removed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        T_i_base = trans_matrix_from_7d_pose(poses[i][0], poses[i][1], poses[i][2],
                                             poses[i][3], poses[i][4], poses[i][5],
                                             poses[i][6])
        pcd_bk_removed.transform(T_i_base)
        multi_view_pcds.append(pcd_bk_removed)
    o3d.visualization.draw_geometries(multi_view_pcds)

    # ----------------------------- cloud registration ----------------------------------
    print('number of pcds:', len(multi_view_pcds))
    print("Full registration ...")
    mcd_coarse = voxel_size * 15
    mcd_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        multi_view_pose_graph = full_registration(multi_view_pcds,
                                                  mcd_coarse,
                                                  mcd_fine)
    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=mcd_fine, edge_prune_threshold=0.25, reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            multi_view_pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
    print("Transform points and display")
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(multi_view_pcds)):
        print(multi_view_pose_graph.nodes[point_id].pose)
        multi_view_pcds[point_id].transform(multi_view_pose_graph.nodes[point_id].pose)
        pcd_combined += multi_view_pcds[point_id]
    print('before down sampling: ', pcd_combined)
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    print('after down sampling: ', pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down], window_name='registered point-clouds')
    o3d.io.write_point_cloud('registered_scene_points.pcd', pcd_combined_down)

    # segment and find clusters of the registered scene point cloud ---------------------------------
    pcd_plane_removed = tt_cloud_plane_removal(pcd_combined_down)
    pcd_clusters = tt_cloud_clustering(pcd_plane_removed, eps=0.015, min_points=50)
    for count, cluster in enumerate(pcd_clusters):
        cleaned_cluster, ind = cluster.remove_radius_outlier(nb_points=2,
                                                             radius=0.005)
        display_inlier_outlier(cluster, ind)
        save_cluster_name = 'cluster' + str(count) + '.pcd'
        o3d.io.write_point_cloud(save_cluster_name, cluster)
    bbx_sizes, bbx_positions = tt_get_cluster_bb_params(pcd_clusters)
    np.savetxt('bbx_sizes.txt', bbx_sizes, fmt='%1.4f')
    np.savetxt('bbx_positions.txt', bbx_positions, fmt='%1.4f')
    print('scene views processing complete!')
    # print(pcd_combined_down)
    # # check point cloud reference frame
    # T_cb = [[-1, 0, 0, 0.7],
    #         [0, 1, 0, -0.16],
    #         [0, 0, -1, 0.5],
    #         [0, 0, 0, 1]]
    # pcd_combined.transform(T_cb)
    # ds = np.linspace(-0.3, 0.3, 21)
    # mx, my = np.meshgrid(ds, ds)
    # xyz = np.zeros((np.size(mx), 3))
    # dxyz = np.ones(np.size(mx))
    # dd = 0.45
    # xyz[:, 0] = np.reshape(mx, -1)
    # xyz[:, 1] = np.reshape(my, -1)
    # xyz[:, 2] = np.reshape(dxyz * dd, -1)
    # # xyz_all = np.copy(xyz)
    # # for i in range(1, 4):
    # #     new_xyz = np.copy(xyz)
    # #     new_xyz[:, 2] = np.reshape(dxyz * dd * i, -1)
    # #     xyz_all = np.append(xyz_all, new_xyz, axis=0)
    # npcd = o3d.geometry.PointCloud()
    # npcd.points = o3d.utility.Vector3dVector(xyz)
    # npcd.paint_uniform_color([1, 0, 0])
    # pcd_combined += npcd
    # o3d.visualization.draw_geometries([pcd_combined])
    # pcd_combined_points = np.asarray(pcd_combined.points)
    # pcd_combined_contact_region = pcd_combined.select_by_index(
    #     np.where(pcd_combined_points[:, 2] < dd)[0])
    # pcd_combined_contact_region.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # pcd_combined_contact_region.orient_normals_consistent_tangent_plane(50)
    # o3d.visualization.draw_geometries([pcd_combined_contact_region],
    #                                   window_name='contact region', point_show_normal=True)
    # silhouette = np.zeros((camera_intrinsic.height, camera_intrinsic.width, 3))
    # fx = camera_intrinsic.intrinsic_matrix[0, 0]
    # fy = camera_intrinsic.intrinsic_matrix[1, 1]
    # cx = camera_intrinsic.intrinsic_matrix[0, 2]
    # cy = camera_intrinsic.intrinsic_matrix[1, 2]
    # # print(fx, fy, cx, cy)
    # pixels = np.empty((0, 2), dtype=np.int)
    # contact_region_points = np.asarray(pcd_combined_contact_region.points)
    # for point in contact_region_points:
    #     u = int(round(fx * point[0] / point[2] + cx))
    #     v = int(round(fy * point[1] / point[2] + cy))
    #     pixels = np.append(pixels, [[v, u]], axis=0)
    #     silhouette[v, u, :] = [255, 255, 255]
    # # print(pixels.shape)
    # # print(np.max(pixels[:, 0]))
    # # print(np.min(pixels[:, 0]))
    # # print(np.max(pixels[:, 1]))
    # # print(np.min(pixels[:, 1]))
    # row_min = np.min(pixels[:, 0])
    # row_max = np.max(pixels[:, 0])
    # col_min = np.min(pixels[:, 1])
    # col_max = np.max(pixels[:, 1])
    # expand_and_erode_rgb(silhouette, pixels, [row_min, row_max, col_min, col_max], 7)
    # cv2.imshow('object silhouette based on grasp direction and depth', silhouette)
    # cv2.imwrite('eroded_projection.jpg', silhouette)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # o3d.visualization.draw_geometries([pcd_combined_down],
    #                                   window_name='registered point-clouds with normal estimation',
    #                                   point_show_normal=True)
