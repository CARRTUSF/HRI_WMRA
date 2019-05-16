import os, sys, requests, io, json
from zipfile import ZipFile
from PIL import Image
import numpy as np, cv2

def upload(url, fpath):
	with open(fpath, 'rb') as f:
		files = {'file' : f}

		try:
			r = requests.post(url, files=files)
			# print(r.text)
			return r.text

		except Exception as e:
			print('Did not send file: {}\nException: {}'.format(fpath, e))

def downloadImg(url):
	r = requests.get(url)
	return Image.open(io.BytesIO(r.content))

def downloadJson(url):
	fname = 'tmp.json'
	r = requests.get(url)
	with open(fname, 'wb') as of:
		of.write(r.content)

	jsonDict = {}
	with open(fname) as inf:
		jsonDict = json.load(inf)
	
	return jsonDict

def downloadZip(url, outDir='./'):
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
		fname = [f for f in flist if f.startswith('vis')][0]
		vis = np.array(Image.open(zf.open(fname)))
		vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
		objDict.update({'vis' : {'mask' : vis, 'fname' : fname, 'url' : urlDict[fname]}})

		# Adds urllist
		# print(urlDict[])

		return objDict

if __name__ == '__main__':
	# Arguments
	url = sys.argv[1]

	# Starts captures
	tmpName = 'tmp-img.png'
	tmpDir = 'tmp'
	width, height = 640, 480

	# Tries realsense first
	try:
		import pyrealsense2 as rs 
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
		config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
		profile = pipeline.start(config)

		while True:
			# print('realsense')
			# Get frames
			frames = pipeline.wait_for_frames()
			frame = np.asanyarray(frames.get_color_frame().get_data())

			# Writes img and uploads
			cv2.imwrite(tmpName, frame)
			retUrl = upload(url, tmpName)
			# print(retUrl)

			# Checks retUrl valid
			if not retUrl or not retUrl.startswith('http://'):
				continue
			# print('debug')

			# Downloads infered stuff
			objDict = downloadZip(retUrl, tmpDir)

			# Shows img
			print('showing img')
			visImg = objDict['vis']['mask']
			visImg = cv2.resize(visImg, (1200, 900))
			cv2.imshow('Inference', visImg)
			k = cv2.waitKey(1)
			if k == 27:
				cv2.destroyAllWindows()
				break 

	except Exception as e:
		print(e)
		cap = cv2.VideoCapture(0)
		while True:
			# Pulls frame
			ret, frame = cap.read()
			if not ret: continue
			# Sends off to server
			# print('debug')
			cv2.imwrite(tmpName, frame)
			retUrl = upload(url, tmpName)
			print(retUrl)

			# Checks retUrl valid
			if not retUrl or not retUrl.startswith('http://'):
				continue
			# print('debug')

			# Downloads infered stuff
			objDict = downloadZip(retUrl, tmpDir)

			# Shows img
			visImg = objDict['vis']['mask']
			cv2.imshow('Inference', visImg)
			k = cv2.waitKey(1)
			if k == 27:
				cv2.destroyAllWindows()
				break 