import rospy
import ros_numpy
from cv2 import cv2
import numpy as np
import open3d as o3d
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
import time

bridge = CvBridge()
color_img = np.zeros((100, 100))
depth_img = np.zeros((100, 100))
xyz = np.empty((0, 3))
rgb = np.empty((0, 3))


def color_callback(color_msg):
    global color_img
    color_img = bridge.imgmsg_to_cv2(color_msg, 'rgb8')


def depth_callback(depth_msg):
    global depth_img
    depth_img = bridge.imgmsg_to_cv2(depth_msg)


def pc2_callback(ros_pc2_msg):
    global xyz, rgb
    cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(ros_pc2_msg)
    rgb_arr = cloud_arr['rgb']
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.float64)/255
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.float64)/255
    b = np.asarray(rgb_arr & 255, dtype=np.float64)/255
    # print(r.shape)
    rgb = np.stack((r, g, b), axis=1).reshape((-1, 3))
    xyz = np.stack((cloud_arr['x'], cloud_arr['y'], cloud_arr['z']), axis=1).reshape((-1, 3))
    # print(rgb.shape)
    # print(xyz.shape)


def save_point_cloud_to_file(file_name, pc_xyz, pc_rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    o3d.io.write_point_cloud(file_name, pcd)
    print('number of points: ', pc_xyz.shape, pc_rgb.shape)
    print('point cloud data saved to: ', file_name)


if __name__ == '__main__':
    # global color_img, depth_img, xyz, rgb
    rospy.init_node('TT_rs_camera_image_recorder', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, color_callback, queue_size=1)
    # rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_callback, queue_size=1)
    rospy.Subscriber('/camera/depth/color/points', PointCloud2, pc2_callback, queue_size=1)
    while not rospy.is_shutdown():
        cv2.imshow('color image', color_img)
        # cv2.imshow('depth image', depth_img)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
        elif key == ord('s'):
            # time = time.time()
            # color_name = str(time) + 'color.png'
            # depth_name = str(time) + 'depth.png'
            # cv2.imwrite(color_name, color_img)
            # cv2.imwrite(depth_name, depth_img)
            # print('Images saved as: ')
            # print(color_name)
            # print(depth_name)
            points_name = 's3.pcd'
            save_point_cloud_to_file(points_name, xyz, rgb)
    cv2.destroyAllWindows()
