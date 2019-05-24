#!/bin/bash
sudo $CONDA_PREFIX/bin/python http_server.py \
	--cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
	--output-dir ./uploads-pipe1 \
	--wts ./models/e2e_mask_rcnn_R-101-FPN_2x.pkl \
	--thresh 0.3 \
	--cuda $1 \
	--ip $2 \
	--port $3
	