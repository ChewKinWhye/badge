#! /bin/sh

#PBS -N CAML
#PBS -l select=1:ncpus=10:mem=80gb:ngpus=1
cd $PBS_O_WORKDIR;


image="/app1/common/singularity-img/hopper/pytorch/pytorch_2.3.0_cuda_12.4_ngc_24.04.sif"

mkdir -p $save_dir
singularity exec --nv --cleanenv $image bash << EOF > $save_dir/output.txt 2> $save_dir/error.txt

source ~/torch_env/bin/activate

python main.py --dataset $dataset --seed $seed --pretrained $pretrained --architecture $architecture --method $method --save_dir $save_dir --alg $alg --inner_steps $inner_steps --inner_lr $inner_lr

EOF
