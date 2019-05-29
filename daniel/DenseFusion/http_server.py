import os, sys, flask, requests, jsonpickle, argparse 

# Argument parser
def parse_args():
	parser = argparse.ArgumentParser()
	parser.ArgumentParser()
	parser.add_argument('-m', '--model', type=str, default = '',  help='resume PoseNet model')
	parser.add_argument('-mr', '--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
	parser.add_argument('-c', '--cuda', type=str, default = '1',  help='Cuda card number to use')
	parser.add_argument('-ip', '--ip', type=str, default = '127.0.0.1',  help='Domain or ip to use')
	parser.add_argument('-p', '--port', type=str, default = '666',  help='Port to use')
	parser.add_argument('-dip', '--dip', type=str, default = '127.0.0.1',  help='Domain or ip for Detectron server')
	parser.add_argument('-dp', '--dport', type=str, default = '665',  help='Port for Detectron server')

	# Needs at least 2 arguments -m and -mr to run
	if(len(sys.argv)) < 3:
		args.print_help()
		sys.exit(1)
	return parser.parse_args()

### GLOBALS ###
args = parse_args()
DOMAIN = args.ip 
PORT = args.port
DET_DOMAIN = args.dip
DET_PORT = args.dport
app = flask.Flask(__name__)
refiner = None
estimator = None

# Transformation stuff
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

# Original params
# cam_cx = 312.9869
# cam_cy = 241.3109
# cam_fx = 1066.778
# cam_fy = 1067.487
# cam_scale = 10000.0

# Intel RealSense D435 Params Color
cam_cx = 327.69
cam_cy = 242.552
cam_fx = 618.2
cam_fy = 618.74
cam_scale = 0.0010000000474974513

# Network/Model Params
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1

# import _init_paths
sys.path.insert(0, os.getcwd())

# Sets up cuda devices
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

# Import network shit
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

# Converts bounding box to their format
def get_bbox(posecnn_rois, idx=None):
	if idx != None:
		rmin = int(posecnn_rois[idx][3]) + 1
		rmax = int(posecnn_rois[idx][5]) - 1
		cmin = int(posecnn_rois[idx][2]) + 1
		cmax = int(posecnn_rois[idx][4]) - 1
	else:
		rmin = int(posecnn_rois[1]) + 1
		rmax = int(posecnn_rois[3]) - 1
		cmin = int(posecnn_rois[0]) + 1
		cmax = int(posecnn_rois[2]) - 1
	r_b = rmax - rmin
	for tt in range(len(border_list)):
		if r_b > border_list[tt] and r_b < border_list[tt + 1]:
			r_b = border_list[tt + 1]
			break
	c_b = cmax - cmin
	for tt in range(len(border_list)):
		if c_b > border_list[tt] and c_b < border_list[tt + 1]:
			c_b = border_list[tt + 1]
			break
	center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
	rmin = center[0] - int(r_b / 2)
	rmax = center[0] + int(r_b / 2)
	cmin = center[1] - int(c_b / 2)
	cmax = center[1] + int(c_b / 2)
	if rmin < 0:
		delt = -rmin
		rmin = 0
		rmax += delt
	if cmin < 0:
		delt = -cmin
		cmin = 0
		cmax += delt
	if rmax > img_width:
		delt = rmax - img_width
		rmax = img_width
		rmin -= delt
	if cmax > img_length:
		delt = cmax - img_length
		cmax = img_length
		cmin -= delt
	return rmin, rmax, cmin, cmax

# Uploads color img to the detectron
def upload(url, frame):
	# Prep headers for http req
	content_type = 'application/json'
	headers = {'content_type': content_type}

	# jsonpickle the numpy frame
	_, frame_png = cv2.imencode('.png', frame)
	frame_json = jsonpickle.encode(frame_png)

	# Post and get response
	try:
		# Gets response and converts it 
		response = requests.post(url, data=frame_json, headers=headers)
		if response.text:
			# Decode response and return it
			retList = jsonpickle.decode(response.text)
			retList[0] = cv2.imdecode(retList[0], cv2.IMREAD_COLOR)
			retList[-1] = [cv2.imdecode(m, cv2.IMREAD_GRAYSCALE) for m in retList[-1]]

			# returns [vis.png, bbList, labelList, scoreList, maskList]
			return retList
		else:
			return None
	except:
		return None

# Converts output to csv string
def createCSV(bbList, labelList, scoreList, poseList):
	# Creates csv string using data form inferences
	csvStr = ''
	for bbs, label, score, pose in zip(bbList, labelList, scoreList, poseList):
		line = [label, str(score)] + [str(f) for f in bbs] + [str(p) for p in pose]
		lineStr = ','.join(line) + '\n'
		csvStr += lineStr

	return csvStr

# Awaits POST requests for RGBd file upload
@app.route('/', methods=['POST'])
def upload_file():
	# Globals
	global refiner, estimator

	# Get request and unzip/decode
	r = flask.request
	imlist = jsonpickle.decode(r.data)
	im, imd = cv2.imdecode(imlist[0], cv2.IMREAD_COLOR), cv2.imdecode(imlist[1], cv2.IMREAD_GRAYSCALE)

	# Send RGB to Detectron Mask R-CNN
	url = f'http://{DET_DOMAIN}:{DET_PORT}'
	# returns [vis.png, bbList, labelList, scoreList, maskList]
	retList = upload(url, im)

	# Starts shit
	_, bbList, labelList, scoreList, maskList = retList
	img = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
	depth = imd
	print('depth:\n', depth[len(depth)/2:len(depth)/2+10, len(depth[0])/2:len(depth[0])/2+10])
	print('max depth:', depth.max())
	my_result_wo_refine = []
	my_result = []
	itemid = 1
	
	# Original Network
	# posecnn_meta = scio.loadmat('mycode/samples/input/000000.mat')
	# label = np.array(posecnn_meta['labels'])
	# posecnn_rois = np.array(posecnn_meta['rois'])
	# lst = posecnn_rois[:, 1:2].flatten()
	# for idx in range(len(lst)):
	# 	itemid = lst[idx]
	# 	# try:
	# 	# cmin, rmin, cmax, rmax = int(posecnn_rois[idx][2]), int(posecnn_rois[idx][3]), int(posecnn_rois[idx][4]), int(posecnn_rois[idx][5])
	# 	rmin, rmax, cmin, cmax = get_bbox(posecnn_rois, idx)
	# 	print(cmin, rmin, cmax, rmax)
	# 	mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
	# 	mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
	# 	mask = mask_label * mask_depth
	
	# Goes through all objects Detectron found
	for bb, mask, score, label in zip(bbList, maskList, scoreList, labelList):
		# cmin, rmin, cmax, rmax = bb
		# print(cmin, rmin, cmax, rmax)
		rmin, rmax, cmin, cmax = get_bbox(bb, None)
		# print(cmin, rmin, cmax, rmax)
		mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
		mask_label = ma.getmaskarray(ma.masked_equal(mask, 1))
		mask = mask_label * mask_depth

		choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
		# print(mask.shape)
		# print(len(choose))
		# for i in range(rmin, rmax):
		# 	for j in range(cmin, cmax):
		# 		val = mask[i,j]
		# 		print(val, end=' ')
		# 	print()
		# print(mask[rmin:rmax, cmin:cmax])
		if len(choose) >= num_points:
			c_mask = np.zeros(len(choose), dtype=int)
			c_mask[:num_points] = 1
			np.random.shuffle(c_mask)
			choose = choose[c_mask.nonzero()]
		else:
			# print(choose)
			choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

		depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
		xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
		ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
		choose = np.array([choose])

		pt2 = depth_masked / cam_scale
		pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
		pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
		cloud = np.concatenate((pt0, pt1, pt2), axis=1)

		img_masked = np.array(img)[:, :, :3]
		img_masked = np.transpose(img_masked, (2, 0, 1))
		img_masked = img_masked[:, rmin:rmax, cmin:cmax]

		cloud = torch.from_numpy(cloud.astype(np.float32))
		choose = torch.LongTensor(choose.astype(np.int32))
		img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
		index = torch.LongTensor([itemid - 1])

		cloud = Variable(cloud).cuda()
		choose = Variable(choose).cuda()
		img_masked = Variable(img_masked).cuda()
		index = Variable(index).cuda()

		# print('DEBUG')
		cloud = cloud.view(1, num_points, 3)
		img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

		pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
		pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

		pred_c = pred_c.view(bs, num_points)
		how_max, which_max = torch.max(pred_c, 1)
		pred_t = pred_t.view(bs * num_points, 1, 3)
		points = cloud.view(bs * num_points, 1, 3)

		my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
		my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
		my_pred = np.append(my_r, my_t)
		my_result_wo_refine.append(my_pred.tolist())

		for ite in range(0, iteration):
			T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
			my_mat = quaternion_matrix(my_r)
			R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
			my_mat[0][3] = my_t[0]
			my_mat[1][3] = my_t[1]
			my_mat[2][3] = my_t[2]
			
			new_cloud = torch.bmm((cloud - T), R).contiguous()
			pred_r, pred_t = refiner(new_cloud, emb, index)
			pred_r = pred_r.view(1, 1, -1)
			pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
			my_r_2 = pred_r.view(-1).cpu().data.numpy()
			my_t_2 = pred_t.view(-1).cpu().data.numpy()
			my_mat_2 = quaternion_matrix(my_r_2)

			my_mat_2[0][3] = my_t_2[0]
			my_mat_2[1][3] = my_t_2[1]
			my_mat_2[2][3] = my_t_2[2]

			my_mat_final = np.dot(my_mat, my_mat_2)
			my_r_final = copy.deepcopy(my_mat_final)
			my_r_final[0][3] = 0
			my_r_final[1][3] = 0
			my_r_final[2][3] = 0
			my_r_final = quaternion_from_matrix(my_r_final, True)
			my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

			my_pred = np.append(my_r_final, my_t_final)
			my_r = my_r_final
			my_t = my_t_final

		my_result.append(my_pred.tolist())
		itemid += 1
		# except ZeroDivisionError:
		# 	# print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
		# 	print('divide by zero error')
		# 	# my_result_wo_refine.append([0.0 for i in range(7)])
		# 	my_result.append([0.0 for i in range(7)])

	# DEBUG
	# print(my_result)

	# Creates return csv
	retStr = createCSV(bbList, labelList, scoreList, my_result)
	
	return retStr

def main():
	# Globals
	global refiner, estimator

	# Sets up refined network
	refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
	refiner.cuda()
	refiner.load_state_dict(torch.load(args.refine_model))
	refiner.eval()

	# Sets up unrefined 
	estimator = PoseNet(num_points = num_points, num_obj = num_obj)
	estimator.cuda()
	estimator.load_state_dict(torch.load(args.model))
	estimator.eval()

	# Runs http server
	# app.run(port=PORT, host='0.0.0.0', debug=True)
	app.run(port=PORT, host=DOMAIN, debug=False)

if __name__ == '__main__':
	main()