import os, sys, numpy as np, cv2 as cv, torch, pyrealsense2 as rs
import _init_paths
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

WIDTH, HEIGHT = 640, 480

# norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1

# Gets bounding box
def get_bbox(posecnn_rois):
	rmin = int(posecnn_rois[idx][3]) + 1
	rmax = int(posecnn_rois[idx][5]) - 1
	cmin = int(posecnn_rois[idx][2]) + 1
	cmax = int(posecnn_rois[idx][4]) - 1
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

# Sets up camera params and starts streaming
def setupRS():
	# Creates pipeline
	pipeline = rs.pipeline()

	# Configs the camera params and enables streams
	config = rs.config()
	config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

	# Align object
	align_to = rs.stream.color 
	align = rs.align(align_to)

	# Starts streaming and returns objects
	pipeline.start(config)
	return (pipeline, align)

# Runs inference on the frames
def streamImgs(pipeline, align, modelPath):
	# estimator = PoseNet(num_points = num_points, num_obj = num_obj)
	# estimator.cuda()
	# estimator.load_state_dict(torch.load(modelPath))
	# estimator.eval()

	flag = True
	while flag:
		try:
			# Pulls aligned frames from stream
			frames = pipeline.wait_for_frames()
			aframes = align.process(frames)
			dframe = np.asanyarray(aframes.get_depth_frame().get_data())
			cframe = np.asanyarray(aframes.get_color_frame().get_data())

			# Visualize
			# catframe = np.hstack((cframe, dframe))
			cv.imshow('real sense', dframe)
			k = cv.waitKey(100)
			if k == 27:
				cv.destroyAllWindows()
				break

			# Runs inference on rgbd
			# pose = runInf(dframe, cframe)


		except Exception as e:
			print('\nStreaming Stopped')
			print('Exception: {}'.format(e))
			flag = False
			pipeline.stop()

# Runs inference on image
def runInf(dframe, cframe):
	pass

if __name__ == '__main__':
	# Arguments
	modelPath = sys.argv[1]

	# Test Camera setup
	pipeline, align = setupRS()
	streamImgs(pipeline, align, modelPath)
	