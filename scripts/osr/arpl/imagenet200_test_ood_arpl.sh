#!/bin/bash
# sh scripts/osr/arpl/imagenet200_test_ood_arpl.sh

#     --network.checkpoint $YOUR_CHECKPOINT_PATH/imagenet200_arpl_net_arpl_e90_lr0.1/s0/best_NetF.ckpt,
#     $YOUR_CHECKPOINT_PATH/imagenet200_arpl_net_arpl_e90_lr0.1/s0/best_criterion.ckpt \

# NOTE!!!!
# need to manually change the checkpoint path in configs/pipelines/test/test_arpl.yml
SCHEME="ood" # "ood" or "fsood"
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_${SCHEME}.yml \
    configs/networks/arpl_net.yml \
    configs/pipelines/test/test_arpl.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --network.feat_extract_network.name resnet18_224x224 \

    --num_workers 8 \
    --evaluator.ood_scheme ${SCHEME} \
    --seed 0
