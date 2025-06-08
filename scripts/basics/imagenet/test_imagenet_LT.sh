#!/bin/bash
# sh scripts/basics/imagenet/test_imagenet_LT.sh

python main.py \
--config configs/datasets/imagenet/imagenet_LT.yml \
configs/networks/resnet18_224x224.yml \
configs/pipelines/test/test_acc.yml \
configs/preprocessors/base_preprocessor.yml \
configs/imb_subset/train_keep_original.yml \
configs/imb_subset/test_keep_original.yml \
--num_workers 20 \
--dataset.test.batch_size 512 \
--dataset.val.batch_size 512 \
--network.pretrained True \
--network.checkpoint "./results/imagenet_LT_resnet18_224x224_base_e100_lr0.1_Original_default/s0/best.ckpt" \
--save_output True \
--num_gpus 1
