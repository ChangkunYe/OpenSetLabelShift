#!/bin/bash
# sh scripts/ood/vim/cifar100_test_ood_vim.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    configs/imb_subset/train_keep_original.yml \
    configs/imb_subset/long_tailed/lt10.yml \
    --num_workers 8 \
    --network.checkpoint '$YOUR_CHECKPOINT_PATH/cifar100_resnet32_owls_e200_lr0.1_Original_default/s0/best.ckpt' \
    --mark 0 \
    --postprocessor.postprocessor_args.dim 256

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data cifar100 \
   --root $YOUR_CHECKPOINT_PATH/cifar100_resnet32_owls_e200_lr0.1_Original_default \
   --postprocessor vim \
   --save-score --save-csv \
   --train_subset_config ./configs/imb_subsetLO/train_keep_original.yml \
   --test_subset_config ./configs/imb_subset/long_tailed/lt10.yml
