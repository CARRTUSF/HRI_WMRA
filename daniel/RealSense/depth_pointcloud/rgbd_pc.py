import os, sys, requests
import numpy as np, cv2, jsonpickle
import pyrealsense2 as rs 

# Uploads to Detectron
def upload(url, color, depth):
	# Prep headers for http req
	content_type = 'application/json'
	headers = {'content_type': content_type}

	# jsonpickle the numpy frame
	_, color_png = cv2.imencode('.png', color)
	_, depth_png = cv2.imencode('.png', depth)
	rgbd_json = jsonpickle.encode([color_png, depth_png])

	# Post and get response
	try:
		response = requests.post(url, data=rgbd_json, headers=headers)
		if response.text:
			# Decode response and return it
			ret_str = response.text
			
			# Returns label, score, bbxmin, bbymin, bbxmax, bbymax, xcenter, ycenter, zcenter, q1, q2, q3, q4
			return ret_str
		else:
			return None
	except:
		return None

def main():
	# Arguments
	if len(sys.argv) < 2:
		domain = '127.0.0.1'
		port = '666'
		outdir = 'test_imgs'
	else:
		domain = sys.argv[1]
		port = sys.argv[2]
		outdir = sys.argv[3]

	# URL for inference
	url = f'http://{domain}:{port}'

	# Makes out dir if needed
	os.system(f'rm -rf {outdir}')
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	# Starts captures
	width, height = 640, 480

	# Sets up realsense
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
	profile = pipeline.start(config)

	# Alignment
	align_to = rs.stream.color
	align = rs.align(align_to)

	# Point cloud stuff
	pc = rs.pointcloud()
	points = rs.points()

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

	# Saves frames and inference
	count = 0
	while count < 10:

		# Get frames and align them
		frames = pipeline.wait_for_frames()
		# frame = np.asanyarray(frames.get_color_frame().get_data())
		aligned_frames = align.process(frames)

		# Get aligned frames
		color_frame = aligned_frames.get_color_frame()
		depth_frame = aligned_frames.get_depth_frame()

		# Check
		if not color_frame or not depth_frame:
			continue

		# Gets imgs
		color_img = np.asanyarray(color_frame.get_data())
		depth_img = np.asanyarray(depth_frame.get_data())
		
		# Sends to detectron
		ret_str = upload(url, color_img, depth_img)
		if not ret_str:
			continue

		# Gets bounding boxes
		lineList = ret_str.strip('\n').split('\n')
		bbList = []
		for line in lineList:
			obj = line.split(',')
			bbs = obj[2:6]
			bbs = [int(bb) for bb in bbs]
			bbList.append(bbs)

		# Draws bounding boxes
		for bb in bbList:
			cv2.rectangle(color_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

		# Sets up filenames
		if count < 10: leadzero = '0'
		else: leadzero = ''
		outpath_color = os.path.join(outdir, f'{leadzero}{count}_color.png')
		outpath_depth = os.path.join(outdir, f'{leadzero}{count}_depth.png')
		outpath_pc = os.path.join(outdir, f'{leadzero}{count}_pc.ply')
		outpath_pca = os.path.join(outdir, f'{leadzero}{count}_pc_aligned.ply')
		outpath_csv = os.path.join(outdir, f'{leadzero}{count}_inf.csv')

		# Writes color and depth
		cv2.imwrite(outpath_color, color_img)
		cv2.imwrite(outpath_depth, depth_img)

		# Write point cloud
		color = frames.get_color_frame()
		depth = frames.get_depth_frame()
		pc.map_to(color)
		points = pc.calculate(depth)
		points.export_to_ply(outpath_pc, color)

		# Writes aligned pc
		pc.map_to(color_frame)
		points = pc.calculate(depth_frame)
		points.export_to_ply(outpath_pca, color_frame)

		# Writes inference csv
		# CSV Format label, score, bbxmin, bbymin, bbxmax, bbymax, xcenter, ycenter, zcenter, q1, q2, q3, q4
		with open(outpath_csv, 'w') as of:
			titles = f'label,score,bbxmin,bbymin,bbxmax,bbymax,xcenter,ycenter,zcenter,q1,q2,q3,q4\n'
			of.write(titles + ret_str)

		# Increments count
		print(f'Frame {leadzero}{count} complete')
		count += 1

if __name__ == '__main__':
	main()
		