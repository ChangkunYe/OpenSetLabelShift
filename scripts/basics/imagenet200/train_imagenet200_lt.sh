#!/bin/bash
# sh scripts/basics/imagenet200/train_imagenet200.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_224x224.yml \
    configs/imb_subset/train/train_lt100.yml \
    configs/pipelines/train/baseline.yml \
    configs/imb_subset/test_keep_original.yml \
    --seed 0

