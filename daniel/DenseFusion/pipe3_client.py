import os, sys, requests, io, json, time
from zipfile import ZipFile
from PIL import Image
import numpy as np, cv2

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

if __name__ == '__main__':
	# Arguments
	url = sys.argv[1]
	imgPath = sys.argv[2]
	depthPath = sys.argv[3]
	outDir = sys.argv[4]

	# Sets proper outDir
	newOutDir = time.strftime("%Y%m%d-%H%M%S")
	outDir = os.path.join(outDir, newOutDir)
	if not os.path.exists(outDir):
		os.makedirs(outDir)

	# Uploads image to Detectron and gets return url for zip file
	retUrl = uploadRGBD(url, imgPath, depthPath)
	# print(retUrl)

	if not retUrl:
		print('Nothing found, please try again.')
		exit(0)

	if retUrl and retUrl.startswith('http://'):
		objDict = downloadZip(retUrl, outDir)
		print(objDict)

	else:
		print('pose:\n{}'.format(retUrl))
		poseCsv = retUrl.split('\n')
		with open(os.path.join(outDir, 'pose.csv'), 'w') as of:
			for line in poseCsv:
				of.write(line + '\n')

	# Checks if it found anything
	# if not retUrl or not retUrl.startswith('http://'):
	# 	print('Did not find any objects, please try again.', retUrl)
		

	# Create output dir if it doesnt exits
	# if not os.path.exists(outDir):
	# 	os.makedirs(outDir)

	# Downloades image info info
	# objDict = downloadZip(retUrl, outDir)