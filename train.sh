#!/bin/bash

# CUDA_VISIBLE_DEVICES=2,3,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py 4
# CUDA_VISIBLE_DEVICES=2,3,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py 4
# CUDA_VISIBLE_DEVICES=2,3,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py 4
# CUDA_VISIBLE_DEVICES=2,3,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py 4
# CUDA_VISIBLE_DEVICES=2,3,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py 4

CUDA_VISIBLE_DEVICES=2,3,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD_classaware.py 4
CUDA_VISIBLE_DEVICES=2,3,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD_cascade.py 4
