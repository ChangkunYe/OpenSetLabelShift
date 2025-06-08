#!/bin/bash
# sh scripts/basics/imagenet200/train_imagenet200.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_mixup.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/imb_subset/keep_original.yml \
    --seed 0 --num_gpus 2 --num_workers 16