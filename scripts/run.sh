#! /bin/sh

walltime=72:00:00
pretrained=1

for alg in conf badge coreset
do
  for method in none mldgc
  do
    for seed in 0 1 2
    do
      for dataset in mcdominoes spuco celeba
      do
        architecture=resnet18
        save_dir=AL_Results/$dataset-$pretrained-$architecture/$alg-$method/$seed
        echo $save_dir
        qsub -l walltime=$walltime -v dataset=$dataset,seed=$seed,pretrained=$pretrained,architecture=$architecture,method=$method,save_dir=$save_dir,alg=$alg submit.sh
      done
      for dataset in multinli civilcomments
      do
        architecture=BERT
        save_dir=AL_Results/$dataset-$pretrained-$architecture/$alg-$method/$seed
        echo $save_dir
        qsub -l walltime=$walltime -v dataset=$dataset,seed=$seed,pretrained=$pretrained,architecture=$architecture,method=$method,save_dir=$save_dir,alg=$alg submit.sh
      done
    done
  done
done
