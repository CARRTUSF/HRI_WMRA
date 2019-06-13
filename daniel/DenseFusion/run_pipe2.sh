#!/bin/bash
sudo sudo /home/anaconda3/envs/pyt1.0.1/bin/python pipe2_desnsefusion.py \
	--model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth \
	--refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth
	