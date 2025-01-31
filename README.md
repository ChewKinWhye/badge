# Cumulative Active Meta-Learning (CAML)
An implementation of the CAML. Details are provided in our paper.

This code was built by modifying [JordanAsh's active learning repository](https://github.com/JordanAsh/badge) and [FacebookResearch's spurious correlation repository](https://github.com/facebookresearch/BalancingGroups).

# Dependencies

To run this code fully, you'll need [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/stable/), [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/), [Pandas](https://pandas.pydata.org/), [Transformers](https://huggingface.co/docs/transformers/index), [Scikit-Learn](https://scikit-learn.org/stable/), and [Scipy](https://scipy.org/).  

### Installation
You can install the required packages using:
```bash
pip install torch torchvision pillow pandas transformers scikit-learn scipy openml
```

To obtain the datasets required to run the code, please refer to [FacebookResearch's spurious correlation repository](https://github.com/facebookresearch/BalancingGroups). The final data directory should look like:


- **data**
  - **mcdominoes**
    - `cifar10`
    - `mnist`
  - **spuco**
    - `spuco_birds`
    - `spuco_dogs`
  - **celeba**
    - `img_align_celeba`
    - `list_eval_partition.csv`
    - `list_attr_celeba.csv`
  - **civilcomments**
    - `all_data_with_identities.csv`
  - **multinli**
    - `cached_train_bert-base-uncased_128_mnli`
    - `cached_dev_bert-base-uncased_128_mnli`
    - `cached_dev_bert-base-uncased_128_mnli-mm`
    - `metadata_random.csv`

# Running an experiment
`python main.py --dataset mcdominoes  --architecture resnet18 --alg BADGE --method none --seed 0`\
runs the BADGE algorithm on the mcdominoes dataset using the resnet18 model, using the default random shuffling of the actively queried data points.

`python main.py --dataset mcdominoes  --architecture resnet18 --alg BADGE --method mldgc --seed 0`\
runs the BADGE algorithm on the mcdominoes dataset using the resnet18 model, using the CAML method to integrate the actively queried data points.

The script to run the full set of experiments can be found in `scripts/`
