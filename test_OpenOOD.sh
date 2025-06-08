#!/bin/bash
ckpt_path='/data2/u6469845/LOGS/Baselines/OpenOOD/cifar100_resnet32_owls_e200_lr0.1_Original_default'
ckpt_lt_path='/data2/u6469845/LOGS/Baselines/OpenOOD/cifar100_resnet32_owls_e200_lr0.1_LT100_default'

test_subset='./configs/imb_subset/long_tailed/lt20.yml'
test_subset2='./configs/imb_subset/long_tailed/lt20r.yml'
# pprocessor='knn'
# pprocessor='openmax'
# pprocessor='mls'
# pprocessor='ash'
pprocessor='react'

python3 scripts/eval_ood.py  --id-data cifar100 \
        --root $ckpt_path \
        --save-score --save-csv \
        --train_subset_config ./configs/imb_subset/train_keep_original.yml \
        --test_subset_config ./configs/imb_subset/test_keep_original.yml \
        --postprocessor $pprocessor

python3 scripts/eval_ood.py  --id-data cifar100 \
        --root $ckpt_path \
        --save-score --save-csv \
        --train_subset_config ./configs/imb_subset/train_keep_original.yml \
        --test_subset_config $test_subset \
        --postprocessor $pprocessor

python3 scripts/eval_ood.py  --id-data cifar100 \
        --root $ckpt_lt_path \
        --save-score --save-csv \
        --train_subset_config ./configs/imb_subset/train/train_lt100.yml \
        --test_subset_config $test_subset \
        --postprocessor $pprocessor

python3 scripts/eval_ood.py  --id-data cifar100 \
        --root $ckpt_lt_path \
        --save-score --save-csv \
        --train_subset_config ./configs/imb_subset/train/train_lt100.yml \
        --test_subset_config $test_subset2 \
        --postprocessor $pprocessor

python3 scripts/eval_ood.py  --id-data cifar100 \
        --root $ckpt_lt_path \
        --save-score --save-csv \
        --train_subset_config ./configs/imb_subset/train/train_lt100.yml \
        --test_subset_config $test_subset2 \
        --postprocessor $pprocessor --ood_ratio 0.1
