#!/bin/bash
# sh scripts/ood/vim/cifar10_test_ood_vim.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    configs/imb_subset/train/train_lt100.yml \
    configs/imb_subset/test_keep_original.yml \
    --num_workers 8 \
    --network.checkpoint '$YOUR_CHECKPOINT_PATH/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
    --mark 0 \
    --postprocessor.postprocessor_args.dim 256

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data cifar10 \
   --root $YOUR_CHECKPOINT_PATH/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor vim \
   --save-score --save-csv \
   --train_subset_config ./configs/imb_subset/train_keep_original.yml \
   --test_subset_config ./configs/imb_subset/test_keep_original.yml
