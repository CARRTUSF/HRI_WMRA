import numpy as np
from cv2 import cv2
from scipy.spatial import KDTree, ConvexHull
from open3d import open3d as o3d
from utils import transformation_matrix_inverse
from tt_pointcloud_utils import tt_generate_plane_cloud_dz, tt_crop_point_cloud, tt_cloud2binary_image
from tqdm import tqdm


def point_in_hull(poi, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], poi) + eq[-1] <= tolerance)
        for eq in hull.equations)


def object_grasp_region_extraction(obj_cloud, _grasp_depth, show_cutting_plane=False):
    if show_cutting_plane:
        cutting_plane = tt_generate_plane_cloud_dz(_grasp_depth)
        obj_with_cutting_plane = obj_cloud + cutting_plane
        o3d.visualization.draw_geometries([obj_with_cutting_plane], 'Object cloud with cutting plane to grasp depth')
    return tt_crop_point_cloud([-1, 1], [-1, 1], [0, _grasp_depth], obj_cloud)


def silhouette_from_point_cloud_projection(pp_image__, pp_pixels__, radius=20):
    pp_image = np.copy(pp_image__)
    init_pixels_kdtree = KDTree(pp_pixels__)
    outliers__ = []
    for pixel in tqdm(pp_pixels__):
        inds = init_pixels_kdtree.query_ball_point(pixel, r=radius)
        nearest_neighbors = []
        for i in inds:
            nearest_neighbors.append(init_pixels_kdtree.data[i])
        nearest_neighbors = np.asarray(nearest_neighbors)
        if nearest_neighbors.shape[0] >= 5:
            neighbor_hull = ConvexHull(nearest_neighbors)
            r_min = np.min(nearest_neighbors[:, 0])
            r_max = np.max(nearest_neighbors[:, 0])
            c_min = np.min(nearest_neighbors[:, 1])
            c_max = np.max(nearest_neighbors[:, 1])
            for row in range(r_min, r_max + 1):
                for col in range(c_min, c_max + 1):
                    if pp_image[row, col] == 0:
                        if point_in_hull((row, col), neighbor_hull):
                            pp_image[row, col] = 1
        else:
            outliers__.append(pixel)
            # pp_image__[pixel[0], pixel[1], :] = [0, 0, 0]
    return np.asarray(pp_image, dtype=np.uint8), outliers__


def tt_obj_pointcloud2silhouette(obj_cloud__, _grasp_depth, silhouette_filling_radius=20,
                                 img_w=1280, img_h=720, f_x=920.8743, f_y=921.56549, c_x=644.76465, c_y=343.153,
                                 show_cutting_plane=False, show_grasp_region_cloud=True, show_results=True,
                                 show_outliers=True, show_proj_points=True):
    obj_contact_region_cloud = object_grasp_region_extraction(obj_cloud__, _grasp_depth, show_cutting_plane)
    if obj_contact_region_cloud.points is not None:
        if show_grasp_region_cloud:
            window_name = 'object contact region for a grasp depth of ' + str(_grasp_depth)
            o3d.visualization.draw_geometries([obj_contact_region_cloud], window_name)

        obj_pp_image, obj_pp_pixels = tt_cloud2binary_image(obj_contact_region_cloud, img_w, img_h, f_x, f_y, c_x, c_y)

        r_min = np.min(obj_pp_pixels[:, 0])
        r_max = np.max(obj_pp_pixels[:, 0])
        c_min = np.min(obj_pp_pixels[:, 1])
        c_max = np.max(obj_pp_pixels[:, 1])
        roi = obj_pp_image[r_min:r_max + 1, c_min:c_max + 1]
        obj_pp_pix_in_roi = obj_pp_pixels - np.array([r_min, c_min])
        print('Extracting object silhouette from object point cloud projection:')
        roi_silhouette, outliers = silhouette_from_point_cloud_projection(roi, obj_pp_pix_in_roi,
                                                                          silhouette_filling_radius)
        if show_results:
            roi_show = np.stack([roi, roi, roi], axis=2) * 255
            roi_silhouette_show = np.stack([roi_silhouette, roi_silhouette, roi_silhouette], axis=2) * 255
            cv2.imshow('object point cloud raw projection', roi_show)
            if show_outliers:
                for outlier in outliers:
                    cv2.circle(roi_silhouette_show, (outlier[1], outlier[0]), 1, (0, 0, 255), -1)
            if show_proj_points:
                for pixel in obj_pp_pix_in_roi:
                    cv2.circle(roi_silhouette_show, (pixel[1], pixel[0]), 1, (255, 0, 0), -1)
            cv2.imshow('object silhouette', roi_silhouette_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        obj_silhouette_img = np.zeros((img_h, img_w))
        obj_silhouette_img[r_min:r_max + 1, c_min:c_max + 1] = roi_silhouette
        return obj_silhouette_img, [[r_min, c_min], [r_max, c_max]]
    else:
        return None, None


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud('cluster0.pcd')
    T_bc = np.asarray([[1, 0, 0, 0.5],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0.5],
                       [0, 0, 0, 1]])
    T_cb = transformation_matrix_inverse(T_bc)
    pcd.transform(T_cb)
    o3d.visualization.draw_geometries([pcd], 'object point-cloud')
    grasp_depth = 0.5
    obj_silhouette, obj_bbx = tt_obj_pointcloud2silhouette(pcd, grasp_depth, silhouette_filling_radius=18,
                                                           show_cutting_plane=True,
                                                           show_grasp_region_cloud=True,
                                                           show_results=True,
                                                           show_outliers=True,
                                                           show_proj_points=True)
    print(obj_bbx)
    print(obj_silhouette.shape)
    print(obj_silhouette.max())
    # cv2.imshow('object silhouette', obj_silhouette)
    # img = np.zeros((720, 1280, 3))
    # img[obj_bbx[0][0]: obj_bbx[1][0]+1, obj_bbx[0][1]: obj_bbx[1][1]+1] = \
    #     np.stack([obj_silhouette, obj_silhouette, obj_silhouette], axis=2) * 255
    # cv2.imshow('full image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

