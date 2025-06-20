#!/bin/bash
# sh scripts/ood/cider/imagenet200_test_cider.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root $YOUR_CHECKPOINT_PATH/imagenet200_cider_net_cider_e10_lr0.01_protom0.95_default \
   --postprocessor cider \
   --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root $YOUR_CHECKPOINT_PATH/imagenet200_cider_net_cider_e10_lr0.01_protom0.95_default \
   --postprocessor cider \
   --save-score --save-csv --fsood
