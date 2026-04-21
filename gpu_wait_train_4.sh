#!/bin/bash

# 定义你要执行的多条命令
COMMANDS=(
    "CUDA_VISIBLE_DEVICES=1,3,5,6 PORT=29501 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_M3FD_15th_2backbone.py 4 --work-dir /home/zhaotianyi/RSDet/work_dirs/debug"
)
# 定义 GPU 内存占用率阈值（百分比），当占用率低于该值时认为 GPU 空闲
THRESHOLD=30

# 持续监测 GPU 状态
while true; do
    CURRENT_TIME=$(date +"%Y-%m-%d %H+8:%M:%S")
    # 获取两个 GPU 的内存占用率
    GPU_MEM_USAGE_1=$(nvidia-smi --id=1 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F ',' '{printf "%.0f\n", ($1 / $2) * 100}')
    GPU_MEM_USAGE_2=$(nvidia-smi --id=3 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F ',' '{printf "%.0f\n", ($1 / $2) * 100}')
    GPU_MEM_USAGE_3=$(nvidia-smi --id=5 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F ',' '{printf "%.0f\n", ($1 / $2) * 100}')
    GPU_MEM_USAGE_4=$(nvidia-smi --id=6 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F ',' '{printf "%.0f\n", ($1 / $2) * 100}')

    # 检查两个 GPU 内存占用率是否都低于阈值
    if (( $(echo "$GPU_MEM_USAGE_1 < $THRESHOLD" | bc -l) )) && (( $(echo "$GPU_MEM_USAGE_2 < $THRESHOLD" | bc -l) ))&& (( $(echo "$GPU_MEM_USAGE_3 < $THRESHOLD" | bc -l) ))&& (( $(echo "$GPU_MEM_USAGE_4 < $THRESHOLD" | bc -l) )); then
        echo "四个 GPU 内存占用率都低于 $THRESHOLD%，开始依次执行命令..."
        for command in "${COMMANDS[@]}"; do
            echo "正在执行命令: $command"
            eval $command
        done
        break
    else
        echo "$CURRENT_TIME 至少有一个 GPU 内存正在使用中，GPU1 占用率: $GPU_MEM_USAGE_1%，GPU3 占用率: $GPU_MEM_USAGE_2%，GPU5 占用率: $GPU_MEM_USAGE_3%，GPU6 占用率: $GPU_MEM_USAGE_4%，等待 60 秒后再次检查..."
        sleep 60
    fi
done
