#!/bin/bash
if [ $# -eq 0 ]
	# No arguments, defaults to GPU=0, 127.0.0.1, 665
	then
		sudo $CONDA_PREFIX/bin/python http_server.py \
			--cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
			--wts ./models/e2e_mask_rcnn_R-101-FPN_2x.pkl \
			--thresh 0.5
	
	# Arguments passed for cuda, ip, port
	else
		sudo $CONDA_PREFIX/bin/python http_server.py \
			--cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
			--wts ./models/e2e_mask_rcnn_R-101-FPN_2x.pkl \
			--thresh 0.3 \
			--cuda $1 \
			--ip $2 \
			--port $3
fi
