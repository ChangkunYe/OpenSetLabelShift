# sh ./scripts/download/dowanload.sh

# download the up-to-date benchmarks and checkpoints
# provided by OpenOOD v1.5
python ./scripts/download/download.py \
	--contents 'checkpoints' \
	--datasets 'ood_v1.5' \
	--checkpoints 'ood_v1.5' \
	--save_dir '$YOUR_CHECKPOINT_PATH' \
	--dataset_mode 'benchmark'