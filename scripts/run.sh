#! /bin/sh

walltime=72:00:00

method=none

architecture=resnet18
pretrained=0
dataset=mcdominoes

for alg in rand conf badge coreset
do
  for seed in 0 1 2
  do
    save_dir=/hpctmp/e0200920/AL_Results/$dataset-$pretrained-$architecture/$alg-$method/$seed

    echo save_dir

    qsub -l walltime=$walltime -v dataset=$dataset,seed=$seed,pretrained=$pretrained,architecture=$architecture,method=$method,save_dir=$save_dir,alg=$alg submit.sh
  done
done