import numpy as np
import argparse
import random
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    # Model Args
    parser.add_argument('--model', help='model - resnet18, resnet50 or vgg', type=str, default='resnet18')
    # Data Args
    parser.add_argument('--data_dir', help='data path', type=str, default='/home/e/e0200920/data')
    parser.add_argument('--dataset', help='dataset, MNIST, FashionMNIST, SVHN, CIFAR10', type=str, default='')
    # Training Args
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=100)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    # Active Learning Args
    parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
    parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=1000)
    parser.add_argument('--nStart', help='number of points to start', type=int, default=1000)
    parser.add_argument('--nEnd', help='total number of points to query', type=int, default=20000)

    # Method Args
    parser.add_argument('--retrain_type', help='retrain the entire classification head or only the scale of the features: [head, scale, none]', type=str, default='none')
    parser.add_argument('--inductive_bias', help='whether to use inductive bias', type=int, default=0)
    parser.add_argument('--bias_weight', help='weight for inductive bias loss', type=float, default=0.5)

    # Random Seed
    parser.add_argument('--seed', help='random seed', type=int, default=0)

    # Save Directory
    parser.add_argument('--save_dir', help='Directory to save trained models', type=str, default='logs')
    args = parser.parse_args()
    print(args)
    return args

def set_seed(seed):
    print("Setting Random Seed")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
