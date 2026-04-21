#!/bin/bash
wait
# 运行你的脚本
TORCH_DISTRIBUTED_DEBUG=DETAIL bash tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_FLIR_14th.py 2 --work-dir ./drebuttal/debug
