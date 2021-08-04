import detectron2
import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

img = cv2.imread('kitchen.jpeg')
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
predictor = DefaultPredictor(cfg)
outputs = predictor(img)
print(outputs['instances'].pred_classes)
print(outputs['instances'].pred_boxes)
v = Visualizer(img[:, :, ::-1])
out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
cv2.imshow('results', out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
