import os, sys, flask, werkzeug as wz, json, requests
from zipfile import ZipFile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('-mr', '--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('-c', '--cuda', type=str, default = '',  help='Cuda card number to use')
parser.add_argument('-ip', '--ip', type=str, default = '',  help='Domain or ip to use')
parser.add_argument('-d', '--dip', type=str, default = '',  help='Domain or ip for Detectron server')
parser.add_argument('-p', '--port', type=int, default = '',  help='Port to use')
opt = parser.parse_args()
if(len(sys.argv)) < 7:
	opt.print_help()
	sys.exit(1)

# Sets up globals
DOMAIN = opt.ip 
PORT = opt.port 
FULLDOMAIN = 'http://{}:{}'.format(DOMAIN, PORT)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tiff', 'bmp'])
UPLOAD_FOLDER = 'uploads-pipe2'
UPLOAD_FOLDER_REL = '/uploads-pipe2/'
app = flask.Flask(__name__)
args = None
refiner = None
estimator = None

# Sets up cuda devices
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=opt.cuda

# import _init_paths
sys.path.insert(0, os.getcwd())

# import os
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

num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1

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

def downloadZip(url, outDir='uploads-pipe2'): 
	if not os.path.exists(outDir):
		os.makedirs(outDir)

	fname = os.path.join(outDir, 'tmp.zip') 
	r = requests.get(url) 
	with open(fname, 'wb') as of: 
		of.write(r.content)

	with ZipFile(fname, 'r') as zf:
		# Gets csv file
		flist = zf.namelist()
		# print(flist)
		for f in flist:
			if f.endswith('csv'):
				csvFile = zf.open(f, 'r')
				csvList = []
				for line in csvFile:
					line = line.decode('utf-8')

					# lineList = [str, float, str, int, int, int, int]
					lineList = line.strip('\n').split(',')
					lineList[1] = float(lineList[1])
					lineList[3:] = [int(f) for f in lineList[3:]]
					csvList.append(lineList)
					# print(csvList[-1])
				csvFile.close()
				break

		# Adds urllist.txt
		urlDict = {}
		with zf.open('urllist.txt') as uf:
			for line in uf:
				line = line.decode('utf-8')
				# print(line)
				urlstr, fname = line.strip('\n').split(',')
				urlDict.update({fname : urlstr})
		# print(urlDict)

		# # Rips images out of zip file
		objDict = {}
		for i, lineList in enumerate(csvList):
			fname, score, label, cmin, rmin, cmax, rmax = lineList
			maskimg = np.array(Image.open(zf.open(fname)))
			# maskimg[maskimg > 0] = 255
			objDict.update({
				i : {
					'score' : score,
					'label' : label,
					'bb' 	: [cmin, rmin, cmax, rmax],
					'mask'  : maskimg,
					'fname' : fname, 
					'url'   : urlDict[fname]
				}
			})

		# Adds visualed image
		# fname = [f for f in flist if f.startswith('vis')][0]
		# vis = np.array(Image.open(zf.open(fname)))
		# vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
		# objDict.update({'vis' : {'mask' : vis, 'fname' : fname, 'url' : urlDict[fname]}})

		# Adds urllist
		# print(urlDict[])
		return objDict

def upload(url, fpath):
	with open(fpath, 'rb') as f:
		files = {'file' : f}

		try:
			r = requests.post(url, files=files)
			# print(r.text)
			return r.text

		except Exception as e:
			print('Did not send file: {}\nException: {}'.format(fpath, e))
			return None

def allowedFile(fname):
	if fname[-3:].lower() in ALLOWED_EXTENSIONS or fname[-4:].lower() in ALLOWED_EXTENSIONS:
		return True
	else:
		return False

def createCSV(objDict, poseList=None):
	# Goes through the stuff.
	lineList = []
	keyList = list(objDict.keys())
	keyList.sort(key=lambda x: int(x))
	for key in keyList:
		obj = objDict[key]
		score, label, bbList = obj['score'], obj['label'], obj['bb']
		line = [str(key), label, str(score)] + [str(f) for f in bbList]
		
		if poseList:
			line += [str(p) for p in poseList[int(key)]]

		lineStr = ','.join(line)
		lineList.append(lineStr)

	return lineList

	# Makes line strings for csv, adds pose if there
	# csvList = []
	# for line in lineList:
	# 	if pose:
	# 		line += [str(f) for ]
	# 	else:
	# 		lineStr = ','.join(line)

def getLists(objDict):
	# Goes through dict
	keyList = list(objDict.keys())
	keyList.sort(key=lambda x: int(x))
	bbList, maskList, scoreList, labelList = [], [], [], []
	for key in keyList:
		obj = objDict[key]
		bbList.append(obj['bb'])
		maskList.append(obj['mask'])
		scoreList.append(obj['score'])
		labelList.append(obj['label'])

	return bbList, maskList, scoreList, labelList

@app.route('/{}/<filename>'.format(UPLOAD_FOLDER))
def uploaded_file(filename):
	return flask.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	global refiner
	if flask.request.method == 'POST':
		file1 = flask.request.files['file1']
		file2 = flask.request.files['file2']
		if file1 and allowedFile(file1.filename) and file2 and allowedFile(file2.filename):
			# Gets filenames, paths, and saves them
			fname1 = wz.secure_filename(file1.filename)
			fpath1 = os.path.join(app.config['UPLOAD_FOLDER'], fname1)
			fname2 = wz.secure_filename(file2.filename)
			fpath2 = os.path.join(app.config['UPLOAD_FOLDER'], fname2)
			# print(fname1, fname2)
			file1.save(fpath1)
			file2.save(fpath2)

			# Gets labels, bbox, and masks
			retUrl = upload(FULLDOMAIN, fpath1)
			objDict = downloadZip(retUrl, UPLOAD_FOLDER)
			
			# DEBUG 1
			# print('objDict: \n', objDict)
			# retUrl = FULLDOMAIN + UPLOAD_FOLDER_REL + 'tmp.zip'
			# return retUrl

			# DEBUG 2
			# retCsv = createCSV(objDict)
			# retStr = str()
			# with open(os.path.join(UPLOAD_FOLDER, 'pose.csv'), 'w') as of:
			# 	for line in retCsv:
			# 		retStr += line + '\n'
			# 		of.write(line + '\n')
			# return retStr

			# Starts shit
			bbList, maskList, scoreList, labelList = getLists(objDict)
			img = Image.open(fpath1)
			depth = np.array(Image.open(fpath2))
			print('depth:\n', depth[:10, :10])
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
			retCsv = createCSV(objDict, my_result)
			retStr = str()
			with open(os.path.join(UPLOAD_FOLDER, 'pose.csv'), 'w') as of:
				for line in retCsv:
					retStr += line + '\n'
					of.write(line + '\n')
			
			# retStr = str()
			# with open(os.path.join(UPLOAD_FOLDER, 'pose.csv'), 'w') as of:
			# 	for line in my_result:
			# 		lineStr = ','.join([str(l) for l in line])
			# 		retStr += ','.join([str(l) for l in line]) + '\n'
			# 		of.write(lineStr + '\n')
			
			return retStr

def main():
	# Globals
	global refiner, estimator

	# Setup upload folder and run server
	if not os.path.exists(UPLOAD_FOLDER):
		os.makedirs(UPLOAD_FOLDER)

	# Sets up network
	refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
	refiner.cuda()
	refiner.load_state_dict(torch.load(opt.refine_model))
	refiner.eval()

	estimator = PoseNet(num_points = num_points, num_obj = num_obj)
	estimator.cuda()
	estimator.load_state_dict(torch.load(opt.model))
	estimator.eval()

	app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
	app.run(port=PORT, host='0.0.0.0', debug=False)

if __name__ == '__main__':
	main()