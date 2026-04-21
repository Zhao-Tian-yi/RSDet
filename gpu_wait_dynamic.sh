#!/bin/bash

# 定义 GPU 内存占用率阈值（百分比），当占用率低于该值时认为 GPU 空闲
THRESHOLD=20
# 需要的空闲 GPU 数量
REQUIRED_FREE_GPUS=6
# 定义你要执行的多条命令
#COMMANDS=(
#    "./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_M3FD_15th_2backbone.py 4 --work-dir /home/zhaotianyi/RSDet/work_dirs/M3FD/4gpu_multianchor_scale3_8_1280_960_mixup"
#    "./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_MFAD_15th_2backbone.py 4 --work-dir /home/zhaotianyi/RSDet/work_dirs/MFAD/4gpu_multianchor_scale3_8_1280_960_nopretrain"
#    "./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_MFAD_15th_2backbone.py 4 --work-dir /home/zhaotianyi/RSDet/work_dirs/MFAD/4gpu_multianchor_scale3_8_1280_960_lr3"
#    "./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_MFAD_15th_2backbone.py 2 --work-dir /home/zhaotianyi/RSDet/work_dirs/MFAD/2gpu_multianchor_scale3_8_1024_768_4"
#    "./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_common_unique_MFAD_15th_2backbone.py 2 --work-dir /home/zhaotianyi/RSDet/work_dirs/MFAD/2gpu_multianchor_scale3_8_1024_768_5"
#)
COMMANDS=(
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py 6
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py 6
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py 6
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py 6
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py 6
)
# 持续监测 GPU 状态
while true; do
    CURRENT_TIME=$(date +"%Y-%m-%d %H+8:%M:%S")
    # 获取所有可用 GPU 的数量
    NUM_GPUS=8
    FREE_GPUS=()
    for ((i = 7; i > 0; i--)); do
        GPU_MEM_USAGE=$(nvidia-smi --id=$i --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F ',' '{printf "%.0f\n", ($1 / $2) * 100}')
        if (( $(echo "$GPU_MEM_USAGE < $THRESHOLD" | bc -l) )); then
            FREE_GPUS+=($i)
        fi
    done

    # 检查是否有足够的空闲 GPU
    if [ ${#FREE_GPUS[@]} -ge $REQUIRED_FREE_GPUS ]; then
        # 取前 4 张空闲的 GPU
        SELECTED_GPUS=("${FREE_GPUS[@]:0:$REQUIRED_FREE_GPUS}")
        # 更严格地拼接以逗号分隔的字符串
        SELECTED_GPUS_STR=""
        for ((j = 0; j < ${#SELECTED_GPUS[@]}; j++)); do
            if [ $j -gt 0 ]; then
                SELECTED_GPUS_STR="${SELECTED_GPUS_STR},"
            fi
            SELECTED_GPUS_STR="${SELECTED_GPUS_STR}${SELECTED_GPUS[$j]}"
        done
        echo $SELECTED_GPUS_STR
        echo "找到 $REQUIRED_FREE_GPUS 张空闲的 GPU: $SELECTED_GPUS_STR，开始依次执行命令..."
        for command in "${COMMANDS[@]}"; do
            full_command="CUDA_VISIBLE_DEVICES=$SELECTED_GPUS_STR $command"
            echo "正在执行命令: $full_command"
            eval $full_command
        done
        break
    else
        echo "$CURRENT_TIME 可用的空闲 GPU 数量不足 $REQUIRED_FREE_GPUS 张，当前空闲 GPU 数量: ${#FREE_GPUS[@]}，等待 20 秒后再次检查..."
        sleep 20
    fi
done