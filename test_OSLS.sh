#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# dataset_name="cifar10"
# ckpt_path="$YOUR_CHECKPOINT_PATH/cifar10_resnet18_32x32_base_e100_lr0.1_default"
#noise_gamma=0.2

dataset_name="cifar100"
ckpt_path="$YOUR_CHECKPOINT_PATH/cifar100_resnet18_32x32_base_e100_lr0.1_default"
noise_gamma=0.1

#dataset_name="imagenet200"
#ckpt_path="$YOUR_CHECKPOINT_PATH/imagenet200_resnet18_224x224_base_e90_lr0.1_default"
#noise_gamma=0.2

#dataset_name="imagenet"
#ckpt_path="$YOUR_CHECKPOINT_PATH/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9"
#noise_gamma=0.2

train_subset="./configs/imb_subset/train_keep_original.yml"

testset_prefix="./configs/imb_subset/long_tailed/"
declare -a arr=("lt100.yml" "lt50.yml" "lt10.yml" "test_keep_original.yml" "lt10r.yml" "lt50r.yml" "lt100r.yml")

# testset_prefix="./configs/imb_subset/dirichlet/"
# declare -a arr=("dir1.yml" "dir10.yml")


for pp in "openmax" "mls" "react" "knn" "ash"
do
 echo $pp
 for ratio in 0.01 0.1 1.0
 do
   echo $ratio
   for path in "${arr[@]}"
   do
     echo "$path"
     python3 scripts/eval_ood.py  --id-data $dataset_name \
             --root $ckpt_path \
             --save-score --save-csv \
             --train_subset_config $train_subset \
             --test_subset_config "$testset_prefix$path" \
             --postprocessor $pp --ood_ratio $ratio \
             --ood_noise_gamma $noise_gamma
   done
 done
done

