# import open3d as o3d
from open3d import open3d as o3d
import numpy as np
from cv2 import cv2
from scipy.spatial import KDTree


def point_cloud_cropping(x_bounds, y_bounds, z_bounds, cloud):
    # *_bounds=[lower bound, higher bound]
    bounding_box = o3d.geometry.AxisAlignedBoundingBox([x_bounds[0], y_bounds[0], z_bounds[0]],
                                                       [x_bounds[1], y_bounds[1], z_bounds[1]])
    return cloud.crop(bounding_box)


def plane_removal_and_clustering(cloud):
    final_clusters = []
    plane_model, inliers = cloud.segment_plane(distance_threshold=0.005,
                                               ransac_n=3,
                                               num_iterations=1000)
    outlier_cloud = cloud.select_by_index(inliers, invert=True)
    clusters = tt_cloud_clustering(outlier_cloud, eps=0.015, min_points=50)
    for cluster in clusters:
        # cleaned_cluster, ind = cluster.remove_statistical_outlier(nb_neighbors=20,
        #                                                           std_ratio=0.5)
        cleaned_cluster, ind = cluster.remove_radius_outlier(nb_neighbors=10,
                                                             radius=0.01)
        # more_clusters = point_cloud_clustering(cleaned_cluster, eps=0.015, min_points=40)
        # for final_cluster in more_clusters:
        #     final_clusters.append(final_cluster)
        final_clusters.append(cleaned_cluster)
    return final_clusters


def tt_cloud_plane_removal(cloud):
    plane_model, inliers = cloud.segment_plane(distance_threshold=0.005,
                                               ransac_n=3,
                                               num_iterations=1000)
    return cloud.select_by_index(inliers, invert=True)


def tt_cloud_clustering(cloud, eps=0.04, min_points=50, print_progress=True):
    clusters = []
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        cluster_labels = np.array(cloud.cluster_dbscan(eps, min_points, print_progress))
    for i in range(cluster_labels.max() + 1):
        cluster_i = cloud.select_by_index(np.where(cluster_labels == i)[0])
        clusters.append(cluster_i)
    return clusters


def tt_get_cluster_bb_params(clusters):
    bb_sizes = []
    bb_positions = []
    for cluster in clusters:
        bb = cluster.get_axis_aligned_bounding_box()
        center = cluster.get_center()
        dx, dy, dz = np.asarray(bb.max_bound) - np.asarray(bb.min_bound)
        bb_sizes.append((dx, dy, dz))
        bb_positions.append((center[0], center[1], center[2]))
    return bb_sizes, bb_positions


def display_inlier_outlier(cloud, in_):
    inlier_cld = cloud.select_by_index(in_)
    outlier_cld = cloud.select_by_index(in_, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cld.paint_uniform_color([1, 0, 0])
    inlier_cld.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cld, outlier_cld])


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud('registered_scene_points.pcd')
    print(pcd)
    o3d.visualization.draw_geometries([pcd])
    pcd_cropped = point_cloud_cropping([0, 0.8], [-0.4, 0.4], [-0.05, 0.7], pcd)
    o3d.visualization.draw_geometries([pcd_cropped])
    pcd_plane_removed = tt_cloud_plane_removal(pcd_cropped)
    o3d.visualization.draw_geometries([pcd_plane_removed])
    pcd_clusters = tt_cloud_clustering(pcd_plane_removed, eps=0.015, min_points=50)
    cluster0 = pcd_clusters[0]
    o3d.visualization.draw_geometries([cluster0])
    print(cluster0)
    cleaned_cluster, ind = cluster0.remove_radius_outlier(nb_points=4,
                                                          radius=0.005)
    display_inlier_outlier(cluster0, ind)
    print(cluster0.get_center())
    print(cleaned_cluster.get_center())

    bbx = cluster0.get_axis_aligned_bounding_box()
    print(bbx)
    l, w, h = np.asarray(bbx.max_bound) - np.asarray(bbx.min_bound)
    print(l, w, h)
    bbx_sizes, bbx_positions = tt_get_cluster_bb_params(pcd_clusters)
    print(bbx_positions)

    bbx_kdtree = KDTree(np.asarray(bbx_positions))
    print(bbx_kdtree)
    nn = bbx_kdtree.query([0.5, 0.5, 0])
    print(nn)
    print(bbx_kdtree.data[nn[1]])
    o3d.visualization.draw_geometries([pcd_clusters[nn[1]]])
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
    #                                          ransac_n=3,
    #                                          num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print("Plane equation: ", a, "x + ", b, "y + ", c, "z + ", d, " = 0")
    # inlier_cloud = pcd.select_by_index(inliers)
    # # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud])
    # o3d.visualization.draw_geometries([outlier_cloud])
    #
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(
    #         outlier_cloud.cluster_dbscan(eps=0.04, min_points=50, print_progress=True))
    #
    # max_label = labels.max()
    # print("point cloud has", max_label, "clusters")
    # print(labels)
    # # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # # colors[labels < 0] = 0
    # # outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    # print('labels test----------------')
    # print(labels.shape)
    # pc_index = np.where(labels == 0)
    # print(pc_index[0])
    # print(np.asarray(outlier_cloud.points).shape)
    # seg1_cloud = outlier_cloud.select_by_index(np.where(labels == 0)[0])
    # o3d.visualization.draw_geometries([seg1_cloud])
    #
    # seg2_cloud = outlier_cloud.select_by_index(np.where(labels == 1)[0])
    # o3d.visualization.draw_geometries([seg2_cloud])
    #
    # cl, ind = seg1_cloud.remove_statistical_outlier(nb_neighbors=30,
    #                                                 std_ratio=1.0)
    #
    # display_inlier_outlier(seg1_cloud, ind)
    # o3d.visualization.draw_geometries([cl])
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(cl.cluster_dbscan(eps=0.04, min_points=50, print_progress=True))
    # print("point cloud has", labels.max()+1, "clusters")
    #
    # cl, ind = seg2_cloud.remove_statistical_outlier(nb_neighbors=30,
    #                                                 std_ratio=1.0)
    # display_inlier_outlier(seg2_cloud, ind)
    # o3d.visualization.draw_geometries([cl])
