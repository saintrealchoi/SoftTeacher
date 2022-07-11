#!/usr/bin/env bash
set -x

TYPE=$1
FOLD=$2
PERCENT=$3
GPUS=$4
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py configs/soft_teacher_visdrone_base_analysis/analysis_2.py --launcher pytorch \
    --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}
