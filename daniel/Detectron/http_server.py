from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, flask, werkzeug as wz, json, zipfile, io
from zipfile import ZipFile
from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import time
import jsonpickle

def parse_args():
	parser = argparse.ArgumentParser(description='End-to-end inference')
	parser.add_argument(
		'--cfg',
		dest='cfg',
		help='cfg model file (/path/to/model_config.yaml)',
		default=None,
		type=str
	)
	parser.add_argument(
		'--wts',
		dest='weights',
		help='weights model file (/path/to/model_weights.pkl)',
		default=None,
		type=str
	)
	parser.add_argument(
		'--output-dir',
		dest='output_dir',
		help='directory for visualization pdfs (default: /tmp/infer_simple)',
		default='/tmp/infer_simple',
		type=str
	)
	parser.add_argument(
		'--thresh',
		dest='thresh',
		help='Threshold for visualizing detections',
		default=0.7,
		type=float
	)
	parser.add_argument(
		'--kp-thresh',
		dest='kp_thresh',
		help='Threshold for visualizing keypoints',
		default=2.0,
		type=float
	)
	parser.add_argument(
		'--cuda',
		dest='cuda',
		help='Enter cuda card number to use as integer',
		default=0,
		type=str
	)
	parser.add_argument(
		'--ip',
		dest='ip',
		help='Server IP',
		default=0,
		type=str
	)
	parser.add_argument(
		'--port',
		dest='port',
		help='Server Port',
		default=0,
		type=str
	)
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()


args = parse_args()
DOMAIN = str(args.ip)
PORT = int(args.port)
FULLDOMAIN = 'http://{}:{}'.format(DOMAIN, PORT)
UPLOAD_FOLDER = 'uploads-pipe1'
UPLOAD_FOLDER_REL = '/uploads-pipe1/'
app = flask.Flask(__name__)

dummy_coco_dataset = None
model = None

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

from caffe2.python import workspace
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

@app.route('/', methods=['POST'])
def upload_file():
	# Get request and unzip/decode
	r = flask.request
	im = jsonpickle.decode(r.data)
	im = cv2.imdecode(im, cv2.IMREAD_COLOR)

	# Run inference
	timers = defaultdict(Timer)
	with c2_utils.NamedCudaScope(0):
		cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
			model, im, None, timers=timers
		)

	# Get results
	cvimg, retVals = vis_utils.vis_one_image_opencv(
		im, 
		cls_boxes,
		cls_segms,
		cls_keyps,
		dataset=dummy_coco_dataset,
		show_class=True,
		show_box=True,
		thresh=args.thresh,
		kp_thresh=args.kp_thresh
	)
	if not retVals:
		print('No retVals\n')
		return flask.Response(response=None)

	# Encodes to png files
	bbList, labelList, scoreList, maskList = retVals
	pngList = [cv2.imencode('.png', m)[1] for m in maskList]
	retList = [cv2.imencode('.png', cvimg)[1], bbList, labelList, scoreList, pngList]

	# Encodes to jsonpickle and sends json
	retList_encoded = jsonpickle.encode(retList)
	return flask.Response(response=retList_encoded, status=200, mimetype='application/json')

@app.route('/{}/<filename>'.format(UPLOAD_FOLDER))
def uploaded_file(filename):
	return flask.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def main():
	# Load network
	global model, dummy_coco_dataset
	# logger = logging.getLogger(__name__)

	merge_cfg_from_file(args.cfg)
	cfg.NUM_GPUS = 1
	# args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
	assert_and_infer_cfg(cache_urls=False)

	assert not cfg.MODEL.RPN_ONLY, \
		'RPN models are not supported'
	assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
		'Models that require precomputed proposals are not supported'

	model = infer_engine.initialize_model_from_cfg(args.weights)
	dummy_coco_dataset = dummy_datasets.get_coco_dataset()
	
	# Setup upload folder and run server
	if not os.path.exists(UPLOAD_FOLDER):
		os.makedirs(UPLOAD_FOLDER)

	app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
	app.run(port=PORT, host='0.0.0.0', debug=False)
	# app.run(port=PORT, host=DOMAIN, debug=False)

if __name__ == '__main__':
	workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
	setup_logging(__name__)
	main()