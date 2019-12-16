#!/bin/bash
python pipe3_client.py \
	http://home.sawyer0.com:666 \
	mycode/samples/input/000001-color.png \
	mycode/samples/input/000001-depth.png \
	outDir

# python pipe3_client.py \
# 	http://home.sawyer0.com:666 \
# 	mycode/kinect-c.png \
# 	mycode/kinect-d.png \
# 	outDir