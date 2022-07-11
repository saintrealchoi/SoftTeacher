#!/usr/bin/env bash
set -x

TYPE=$1
FOLD=$2
PERCENT=$3
GPUS=$4
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py configs/soft_teacher_coco_gaussian_jitter_sahi/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_gaussian_wo_jitter_sahi_xpre_v2.py --launcher pytorch \
    --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}
