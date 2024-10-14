#! /bin/sh

#PBS -q volta_gpu
#PBS -j oe
#PBS -N pytorch
#PBS -l select=1:ncpus=10:mem=80gb:ngpus=1
#PBS -P 11002407


cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);
image="/app1/common/singularity-img/3.0.0/pytorch_1.12_cuda_11.6.2_cudnn8-py38.sif"

mkdir -p $output_dir

singularity exec $image bash << EOF > $output_dir/output.txt 2> $output_dir/error.txt
export PYTHONPATH=$PYTHONPATH:/home/svu/e0200920/volta_pypkg/lib/python3.8/site-packages

python main.py --dataset $dataset --seed $seed --pretrained $pretrained --architecture $architecture --method $method --save_dir $save_dir --alg $alg

EOF