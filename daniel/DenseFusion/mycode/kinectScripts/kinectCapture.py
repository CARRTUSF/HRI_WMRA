import freenect, cv2, numpy as np

def get_depth():
	# Gets depth
	depth = freenect.sync_get_depth()[0]
	depth *= 2**5

	# Gets pretty depth
	# pdepth = np.copy(depth)
	# np.clip(pdepth, 0, 2**10 - 1, pdepth)
	# pdepth >>= 2
	# pdepth = pdepth.astype(np.uint8)

	return depth

def get_bgr():
	# Pulls frame from video and converts to bgr
	video = freenect.sync_get_video()[0]
	return video[:, :, ::-1]

# Main function for testing
if __name__ == '__main__':
	# Testing
	count = 5

	# Takes count pics
	while count > 0:
		# DEBUG
		print('count =', count)

		# Gets depth and bgr
		depthImg = get_depth()
		# print(f'depth dtype = {depthImg.dtype}')
		# print(f'depth max = {depthImg.max()}, depth min = {depthImg.min()}')
		colorImg = get_bgr()

		# Shows imgs
		# cv2.imshow('Depth', depthImg)
		# cv2.imshow('Color', colorImg)
		# if cv2.waitKey(0) == 27:
		# 	cv2.destroyAllWindows()

		# Write imgs
		cv2.imwrite('{}-depth.png'.format(count), depthImg)
		cv2.imwrite('{}-color.png'.format(count), colorImg)

		# Decrements count
		count -= 1