#!/usr/bin/env bash

TORCH_DISTRIBUTED_DEBUG=DETAIL bash tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_FLIR_14th.py 2 --work-dir ./drebuttal/ablation_dfs
TORCH_DISTRIBUTED_DEBUG=DETAIL bash tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_FLIR_14th_nomiloss.py 2 --work-dir ./drebuttal/ablation_miloss
TORCH_DISTRIBUTED_DEBUG=DETAIL bash tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_FLIR_14th_miloss_conf.py 2 --work-dir ./drebuttal/ablation_miloss_conf
