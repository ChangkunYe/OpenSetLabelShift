#!/bin/bash
# sh scripts/basics/cifar100/train_cifar100.sh

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    configs/imb_subset/train/train_lt100.yml \
    configs/imb_subset/test_keep_original.yml \
    --seed 1