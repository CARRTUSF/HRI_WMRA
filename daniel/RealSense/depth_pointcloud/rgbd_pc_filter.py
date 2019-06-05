import os, sys, requests
import numpy as np, cv2, jsonpickle
import pyrealsense2 as rs

# Intel RealSense D435 Params Color
cam_cx = None
cam_cy = None
cam_fx = None
cam_fy = None
cam_scale = None

def main():
	# Globals
	global cam_cx, cam_cy, cam_fx, cam_fy, cam_scale

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
	width, height, fps = 848, 480, 60

	# Sets up realsense
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
	profile = pipeline.start(config)

	# Get intrinsics
	depth_sensor = profile.get_device().first_depth_sensor()
	cam_scale = depth_sensor.get_depth_scale()
	intrc =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
	cam_cx = intrc.ppx
	cam_cy = intrc.ppy
	cam_fx = intrc.fx
	cam_fy = intrc.fy
	# intrd =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
	# print('Depth Scale: {}'.format(cam_scale))
	# print('Color: {}\nDepth: {}\n'.format(intrc, intrd))
	print(cam_cx, cam_cy, cam_fx, cam_fy, cam_scale)
	# sys.exit(0)

	# Alignment
	align_to = rs.stream.color
	align = rs.align(align_to)

	# Point cloud stuff
	pc = rs.pointcloud()
	points = rs.points()

	# First few images are no good
	count = 0
	while count < 10:
		# Get frames
		frames = pipeline.wait_for_frames()
		count += 1

	# Saves frames and inference
	count = 0
	num_frames = 6
	num_len = len(str(num_frames))
	while count < num_frames:

		# Get frames and align them
		frames = pipeline.wait_for_frames()
		# frame = np.asanyarray(frames.get_color_frame().get_data())
		aligned_frames = align.process(frames)

		# Get aligned frames
		color_frame = aligned_frames.get_color_frame()
		depth_frame = aligned_frames.get_depth_frame()
		depth_frame_filter = aligned_frames.get_depth_frame()

		# Check
		if not color_frame or not depth_frame:
			continue

		# Sets up post processing filters
		decimation_filter = rs.decimation_filter()
		threshold_filter = rs.threshold_filter()
		spatial_filter = rs.spatial_filter()
		temporal_filter = rs.temporal_filter()
		hole_filling_filter = rs.hole_filling_filter()

		# Apply filters
		# depth_frame_filter = decimation_filter.process(depth_frame_filter)
		depth_frame_filter = threshold_filter.process(depth_frame_filter)
		depth_frame_filter = spatial_filter.process(depth_frame_filter)
		depth_frame_filter = temporal_filter.process(depth_frame_filter)
		# depth_frame_filter = hole_filling_filter.process(depth_frame_filter)

		# Gets imgs
		color_img = np.asanyarray(color_frame.get_data())
		depth_img = np.asanyarray(depth_frame.get_data())
		depth_img_filter = np.asanyarray(depth_frame_filter.get_data())
		# print(depth_img.shape, depth_img_filter.shape)

		# For viewability, dont send to network
		depth_img *= 10
		depth_img_filter *= 10
		
		# Sets up filenames
		outpath_color = os.path.join(outdir, f'{str(count).zfill(num_len)}_color.png')
		outpath_depth = os.path.join(outdir, f'{str(count).zfill(num_len)}_depth.png')
		outpath_pc = os.path.join(outdir, f'{str(count).zfill(num_len)}_pc.ply')
		outpath_pca = os.path.join(outdir, f'{str(count).zfill(num_len)}_pc_aligned.ply')
		outpath_depth_filter = os.path.join(outdir, f'{str(count).zfill(num_len)}_depth_filter.png')

		# Writes color and depth
		cv2.imwrite(outpath_color, color_img)
		cv2.imwrite(outpath_depth, depth_img)
		cv2.imwrite(outpath_depth_filter, depth_img_filter)

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

		# Increments count
		print(f'Frame:{str(count).zfill(num_len)} complete')
		count += 1

# If main
if __name__ == '__main__':
	main()
		