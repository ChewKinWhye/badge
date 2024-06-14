#! /bin/sh

# Fixed Parameters for testing
model=resnet18
dataset=MNIST
alg=rand
seed=0
bias_weight=0.5

for retrain_type in head scale none
do
	save_dir=/home/e/e0200920/active_learning/$model-$dataset-$alg-$retrain_type-$seed

	echo $save_dir

	sbatch slurm.sh --model model --dataset dataset --alg alg --retrain_type $retrain_type --bias_weight $bias_weight --seed $seed --save_dir $save_dir

done
