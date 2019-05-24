import os, sys, requests, io, json, zipfile
from zipfile import ZipFile
from PIL import Image
import numpy as np, cv2, jsonpickle

# def upload(url, fpath):
# 	with open(fpath, 'rb') as f:
# 		files = {'file' : f}

# 		try:
# 			r = requests.post(url, files=files)
# 			# print(r.text)
# 			return r.text

# 		except Exception as e:
# 			print('Did not send file: {}\nException: {}'.format(fpath, e))

def upload(url, frame):
	# Prep headers for http req
	content_type = 'application/json'
	headers = {'content_type': content_type}

	# jsonpickle the numpy frame
	_, frame_png = cv2.imencode('.png', frame)
	frame_json = jsonpickle.encode(frame_png)

	# Post and get response
	try:
		response = requests.post(url, data=frame_json, headers=headers)
		if response.text:
			# Decode response and return it
			retList = jsonpickle.decode(response.text)
			retList[0] = cv2.imdecode(retList[0], cv2.IMREAD_COLOR)
			retList[-1] = [cv2.imdecode(m, cv2.IMREAD_GRAYSCALE) for m in retList[-1]]
			return retList
		else:
			return None
	except:
		return None

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

if __name__ == '__main__':
	# Arguments
	domain = sys.argv[1]
	port = sys.argv[2]
	url = f'http://{domain}:{port}'

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

		count = 0
		while True:
			# print('realsense')
			# Get frames
			frames = pipeline.wait_for_frames()
			frame = np.asanyarray(frames.get_color_frame().get_data())
			retList = upload(url, frame)
			if not retList:
				continue

			# Writes img and uploads
			# cv2.imwrite(tmpName, frame)
			# retUrl = upload(url, tmpName)
			# print(retUrl)

			# # Checks retUrl valid
			# if not retUrl or not retUrl.startswith('http://'):
			# 	continue
			# # print('debug')

			# # Downloads infered stuff
			# objDict = downloadZip(retUrl, tmpDir)

			# Shows img
			count += 1
			print(f'showing img {count}')
			visImg = retList[0]
			visImg = cv2.resize(visImg, (1200, 900))
			cv2.imshow('Inference', visImg)
			k = cv2.waitKey(1)
			if k == 27:
				cv2.destroyAllWindows()
				break 

	except Exception as e:
		print(e)
		