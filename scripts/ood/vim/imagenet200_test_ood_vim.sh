#!/bin/bash
# sh scripts/ood/vim/imagenet200_test_ood_vim.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root /data2/u6469845/LOGS/Baselines/OpenOOD/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor vim \
    --save-score --save-csv  \
    --train_subset_config ./configs/imb_subset/train_keep_original.yml \
    --test_subset_config ./configs/imb_subset/test_keep_original.yml

# full-spectrum ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root /data2/u6469845/LOGS/Baselines/OpenOOD/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor vim \
    --save-score --save-csv --fsood \
    --train_subset_config ./configs/imb_subset/train_keep_original.yml \
    --test_subset_config ./configs/imb_subset/test_keep_original.yml
