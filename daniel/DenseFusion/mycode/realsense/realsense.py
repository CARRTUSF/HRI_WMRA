import numpy as np, pyrealsense2 as rs2, cv2
import os, sys, requests, io, json, time
from zipfile import ZipFile
from PIL import Image

def uploadRGBD(url, imgPath, depthPath):
	with open(imgPath, 'rb') as f1, open(depthPath, 'rb') as f2:
		files = {'file1' : f1, 'file2' : f2}

		try:
			r = requests.post(url, files=files)
			# print(r.text)
			return r.text

		except Exception as e:
			print('Did not send file: {}\nException: {}'.format(fpath, e))

def downloadZip(url, outDir): 
	# Setup upload folder and run server
	fname = os.path.join(outDir, 'tmp.zip') 
	if not os.path.exists(outDir):
		os.makedirs(outDir)

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
		fname = [f for f in flist if f.startswith('vis')][0]
		vis = np.array(Image.open(zf.open(fname)))
		vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
		objDict.update({'vis' : {'mask' : vis, 'fname' : fname, 'url' : urlDict[fname]}})

		# Adds urllist
		# print(urlDict[])
		return objDict

def pipeconfig(width=640, height=480):
	config = rs2.config()
	config.enable_stream(rs2.stream.depth, width, height, rs2.format.z16, 30)
	config.enable_stream(rs2.stream.color, width, height, rs2.format.bgr8, 30)
	return config

# Main for testing
if __name__ == '__main__':
	# Creates pipeline and config
	pipeline = rs2.pipeline()
	config = pipeconfig(640, 480)
	profile = pipeline.start(config)

	# Gets depth and scale
	depth_sensor = profile.get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()

	# Get intrinsics
	intrc =  profile.get_stream(rs2.stream.color).as_video_stream_profile().get_intrinsics()
	intrd =  profile.get_stream(rs2.stream.depth).as_video_stream_profile().get_intrinsics()
	
	# Camera specs
	print('Depth Scale: {}'.format(depth_scale))
	print('Color: {}\nDepth: {}\n'.format(intrc, intrd))

	# Alignment
	align_to = rs2.stream.color
	align = rs2.align(align_to)

	# First few images are no good
	count = 10
	while count > 0:
		# Get frames
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)

		# Get aligned frames
		depth_frame = aligned_frames.get_depth_frame()
		color_frame = aligned_frames.get_color_frame()

		count -= 1

	# Loops until KeyboardInterupt
	count = 5
	# while True:
	while count > 0:
		# Get frames
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)

		# Get aligned frames
		depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
		color_frame = aligned_frames.get_color_frame()

		# Checks if it got the depth and color frame
		if depth_frame and color_frame:
			# Creates np arrays of img
			depthImg = np.asanyarray(depth_frame.get_data())
			colorImg = np.asanyarray(color_frame.get_data())
			x, y = np.where(depthImg == depthImg.max())[0][0], np.where(depthImg == depthImg.max())[1][0]
			print(x, y)
			print(depthImg.max(), depthImg[x, y], depthImg[x, y] * depth_scale)

			bc = 0
			for x in depthImg.flatten():
				if x > 50000:
					bc += 1
			print(bc, end='\n\n')

			# Wrties frames
			# depthImg3d = np.dstack((depthImg, depthImg, depthImg))
			# depthImg = cv2.applyColorMap(cv2.convertScaleAbs(depthImg3d, alpha=0.03), cv2.COLORMAP_JET)
			cv2.imwrite('{}-depth.png'.format(count), depthImg)
			cv2.imwrite('{}-color.png'.format(count), colorImg)

			# Views frames
			# d = np.dstack((d, d, d))
			# d = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET)
			# images = np.hstack((c, d))
			# cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
			# cv2.imshow('Align Example', images)
			# key = cv2.waitKey(1)
			# # Press esc or 'q' to close the image window
			# if key & 0xFF == ord('q') or key == 27:
			# 	cv2.destroyAllWindows()
			# 	break

			# Decrements counter
			count -= 1