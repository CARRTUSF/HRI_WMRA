"""
This is a ROS node that subscribes to an image topic, and publishes the image object-detection/segmentation result.
Method: detectron2 (https://github.com/facebookresearch/detectron2)
"""
import rospy
import sensor_msgs.msg
import argparse
import cv2
import time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from cv_bridge import CvBridge

INPUT_ROS_IMAGE = sensor_msgs.msg.Image()


def in_img_callback(img_msg):
    global INPUT_ROS_IMAGE
    INPUT_ROS_IMAGE = img_msg


def crop_image2roi(roi, image):
    height = image.shape[0]
    width = image.shape[1]
    roi_h, roi_w = roi
    x1 = int((width - roi_w) / 2)
    y1 = int((height - roi_h) / 2)
    if x1 < 0 or y1 < 0:
        raise Exception('ROI is larger than the image')
    else:
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        return image[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description='Detectron2 ROS node for image-based object recognition/segmentation.')
    parser.add_argument('--input', type=str, help='Input ROS image topic', default='/camera/color/image_raw')
    parser.add_argument('--roi_width', type=int, help='Image region of interest width', default=640)
    parser.add_argument('--roi_height', type=int, help='Image region of interest height', default=480)
    args = parser.parse_args()
    input_img_topic = args.input
    roi = (args.roi_height, args.roi_width)

    rospy.init_node('detectron2_node', anonymous=True)
    print('detectron2_node node initialized, subscribing to topic:', input_img_topic)
    rospy.Subscriber(input_img_topic, sensor_msgs.msg.Image, callback=in_img_callback)
    # pub = rospy.Publisher()
    # prepare tools
    # cvbridge for converting ros image to cv image
    br = CvBridge()
    # detectron2 predictor set up
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    predictor = DefaultPredictor(cfg)
    # main loop
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():

        if INPUT_ROS_IMAGE.header.frame_id != '':
            # convert ros image to cv2 image
            in_cv_img = br.imgmsg_to_cv2(INPUT_ROS_IMAGE, 'bgr8')
            input_image = crop_image2roi(roi, in_cv_img)
            # print('input image size:', input_image.shape)
            cv2.imshow('image', input_image)
            s_time = time.time()
            outputs = predictor(input_image)
            print('mask detection takes:', time.time()-s_time, 'sec')
            v = Visualizer(input_image[:, :, ::-1])
            out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            cv2.imshow('results', out.get_image()[:, :, ::-1])
            cv2.waitKey(5)

        rate.sleep()
        

if __name__ == '__main__':
    main()
