#! /bin/sh

walltime=10:00:00
pretrained=1

for alg in badge
do
  for method in meta_reweight_ANIL
  do
    for seed in 0
    do
      for dataset in mcdominoes
      do
	for inner_steps in 1
	do
	  for inner_lr in 1e-4
	  do
            architecture=resnet18
            save_dir=AL_Results/$dataset-$pretrained-$architecture/$alg-$method-$inner_steps-$inner_lr/$seed
            echo $save_dir
            qsub -l walltime=$walltime -v dataset=$dataset,seed=$seed,pretrained=$pretrained,architecture=$architecture,method=$method,save_dir=$save_dir,alg=$alg,inner_steps=$inner_steps,inner_lr=$inner_lr submit_subset.sh
      	  done
	done
      done
    done
  done
done
