#!/bin/bash
# sh scripts/basics/imagenet/test_imagenet.sh

export CUDA_VISIBLE_DEVICES=4,5,6,7
python main.py \
--config configs/datasets/imagenet/imagenet_LT.yml \
configs/networks/resnet50.yml \
configs/pipelines/train/baseline.yml \
configs/preprocessors/base_preprocessor.yml \
configs/imb_subset/train_keep_original.yml \
configs/imb_subset/test_keep_original.yml \
--num_workers 16 \
--dataset.train.batch_size 64 \
--dataset.test.batch_size 64 \
--dataset.val.batch_size 64 \
--num_gpus 4