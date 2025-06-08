#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# dataset_name="cifar10"
# ckpt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/cifar10_resnet18_32x32_base_e100_lr0.1_default"
# ckpt_lt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/cifar10_resnet18_32x32_base_e100_lr0.1_LT100_default"
#noise_gamma=0.2

dataset_name="cifar100"
ckpt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/cifar100_resnet18_32x32_base_e100_lr0.1_default"
ckpt_lt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/cifar100_resnet18_32x32_base_e100_lr0.1_LT100_default"

#ckpt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/cifar100_resnet32_owls_e200_lr0.1_Original_default"
#ckpt_lt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/cifar100_resnet32_owls_e200_lr0.1_LT100_default"

#dataset_name="imagenet200"
#ckpt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/imagenet200_resnet18_224x224_base_e90_lr0.1_default"
#ckpt_lt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/imagenet200_resnet18_224x224_base_e90_lr0.1_LT100_default"
#noise_gamma=0.2

#dataset_name="imagenet"
#ckpt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9"
#ckpt_lt_path="/data2/u6469845/LOGS/Baselines/OpenOOD/imagenet_LT_resnet50_base_e100_lr0.1_Original_default"
#noise_gamma=0.2

train_subset0="./configs/imb_subset/train_keep_original.yml"
train_subset1="./configs/imb_subset/train/train_lt100.yml"

testset_prefix="./configs/imb_subset/long_tailed/"
declare -a arr=("lt100.yml" "lt50.yml" "lt10.yml" "test_keep_original.yml" "lt10r.yml" "lt50r.yml" "lt100r.yml")

# testset_prefix="./configs/imb_subset/dirichlet/"
# declare -a arr=("dir1.yml" "dir10.yml")

# pprocessor="knn"
pprocessor="openmax"
# pprocessor="mls"
# pprocessor="ash"
# pprocessor="react"

noise_gamma=0.1
# for pp in "openmax" "mls" "react" "knn" "ash"
for gamma in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  echo $gamma
  for ratio in 0.01 0.1 1.0
  do
    echo $ratio
    for path in "${arr[@]}"
    do
      echo "$path"
      python3 scripts/eval_ood.py  --id-data $dataset_name \
              --root $ckpt_path \
              --save-score --save-csv \
              --train_subset_config $train_subset0 \
              --test_subset_config "$testset_prefix$path" \
              --postprocessor $pprocessor --ood_ratio $ratio \
              --ood_noise_gamma $gamma
    done
  done
done


#if [ "$dataset_name" = "imagenet" ]; then
#dataset_name="imagenet_lt"
#train_subset1=$train_subset0
#fi

#for pp in "openmax" "mls" "react" "knn" "ash"
#do
#  echo $pp
#  for ratio in 0.01 0.1 1.0
#  do
#    echo $ratio
#    for path in "${arr[@]}"
#    do
#      echo "$path"
#      python3 scripts/eval_ood.py  --id-data $dataset_name \
#              --root $ckpt_lt_path \
#              --save-score --save-csv \
#              --train_subset_config $train_subset1 \
#              --test_subset_config "$testset_prefix$path" \
#              --postprocessor $pp --ood_ratio $ratio \
#              --ood_noise_gamma $noise_gamma
#    done
#  done
#done
