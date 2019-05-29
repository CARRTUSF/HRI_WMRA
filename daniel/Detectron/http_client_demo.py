import os, sys, requests
import numpy as np, cv2, jsonpickle

# Uploads to Detectron
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
			
			# returns [vis.png, bbList, labelList, scoreList, maskList]
			return retList
		else:
			return None
	except:
		return None

def main():
	# Arguments
	if len(sys.argv) < 2:
		domain = '127.0.0.1'
		port = '665'
	else:
		domain = sys.argv[1]
		port = sys.argv[2]
	url = f'http://{domain}:{port}'

	# Starts captures
	width, height = 640, 480
	try:
		import pyrealsense2 as rs 
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
		config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
		profile = pipeline.start(config)

		while True:
			# Get frames
			frames = pipeline.wait_for_frames()
			frame = np.asanyarray(frames.get_color_frame().get_data())
			
			# Sends to detectron
			retList = upload(url, frame)
			if not retList:
				continue

			# Shows img
			visImg = retList[0]
			visImg = cv2.resize(visImg, (1200, 900))
			cv2.imshow('Inference', visImg)
			k = cv2.waitKey(1)
			if k == 27:
				cv2.destroyAllWindows()
				break 

	except Exception as e:
		print(e)

if __name__ == '__main__':
	main()
		