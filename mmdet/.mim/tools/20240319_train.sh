#!/usr/bin/env bash

TORCH_DISTRIBUTED_DEBUG=DETAIL bash tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_LLVIP_14th.py 2 --work-dir ./rebuttal/LLVIP_ablation_DFS_mil0ss_0001danny
TORCH_DISTRIBUTED_DEBUG=DETAIL bash tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_FLIR_15th.py 2 --work-dir ./rebuttal/FLIR_ablation_DFS_mil0ss_0001danny