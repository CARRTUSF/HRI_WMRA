#!/bin/bash
if [ $# -eq 0 ]
# No arguments, defaults to GPU=0, 127.0.0.1, 665
then
	sudo $CONDA_PREFIX/bin/python http_server.py \
		--model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth \
		--refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth

# Arguments passed for cuda, ip, port
else
	if [ $# -eq 3 ]
	then 
		sudo $CONDA_PREFIX/bin/python http_server.py \
			--model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth \
			--refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth \
			--cuda $1 \
			--ip $2 \
			--port $3
	else
		sudo $CONDA_PREFIX/bin/python http_server.py \
			--model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth \
			--refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth \
			--cuda $1 \
			--ip $2 \
			--port $3 \
			--dip $4 \
			--dport $5
	fi
fi
