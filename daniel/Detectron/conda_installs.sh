#!/bin/bash
conda install -c pytorch pytorch
conda install pyyaml=3.12 matplotlib cython mock scipy six future protobuf flask werkzeug opencv requests
conda install -c conda-forge pycocotools pillow torchvision
make
