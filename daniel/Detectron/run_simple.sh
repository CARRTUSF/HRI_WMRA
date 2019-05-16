#!/bin/bash
python tools/infer_simple.py \
	--cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
	--output-dir ./output \
	--image-ext jpg \
	--wts ./models/e2e_mask_rcnn_R-101-FPN_2x.pkl \
	--im-or-folder ./images/dog.jpg \
	--always-out \
	--output-ext png 