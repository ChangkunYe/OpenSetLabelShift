#!/bin/bash
# sh scripts/uncertainty/deepaugment/imagenet200_test_ood_msp.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root $YOUR_CHECKPOINT_PATH/imagenet200_resnet18_224x224_base_e30_lr0.1_deepaugment \
    --postprocessor msp \
    --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root $YOUR_CHECKPOINT_PATH/imagenet200_resnet18_224x224_base_e30_lr0.1_deepaugment \
    --postprocessor msp \
    --save-score --save-csv --fsood
