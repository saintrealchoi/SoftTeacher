#!/usr/bin/env bash
set -x

TYPE=$1
FOLD=$2
PERCENT=$3
GPUS=$4
PORT=${PORT:-29500}

export unsup_start_iter=5000
export unsup_warmup_iter=2000
export distill_start_iter=15000
export distill_warmup=5000

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py --launcher pytorch \
    --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}