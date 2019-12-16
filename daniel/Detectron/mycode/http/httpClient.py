import os, sys, requests, io
from PIL import Image

def upload(url, fpath):
	with open(fpath, 'rb') as f:
		files = {'file' : f}

		try:
			r = requests.post(url, files=files)
			# print(r.text)
			return r.text

		except Exception as e:
			print('Did not send file: {}\nException: {}'.format(fpath, e))

def download(url):
	r = requests.get(url)
	return Image.open(io.BytesIO(r.content))

if __name__ == '__main__':
	url = sys.argv[1]
	fpath = sys.argv[2]
	
	retUrl = upload(url, fpath)
	print(retUrl)

	img = download(retUrl)
	with open('dl.jpg', 'wb') as fo:
		img.save(fo)