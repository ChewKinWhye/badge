import numpy as np
import argparse
import random
import torch
import tqdm
import torchvision
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import os

def parse_args():
    parser = argparse.ArgumentParser()
    # Model Args
    parser.add_argument('--architecture', help='model - resnet18, resnet50', type=str, default='resnet18')
    parser.add_argument("--pretrained", type=int, default=0, help="Use pretrained model")
    # Data Args
    parser.add_argument('--data_dir', help='data path', type=str, default='/hpctmp/e0200920')
    parser.add_argument('--dataset', help='dataset, mcdominoes, spawrious, spuco, treeperson', type=str, default='mcdominoes')
    parser.add_argument("--spurious_strength", type=float, default=0.95, help="Strength of spurious correlation")
    # Training Args
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--num_epochs', help='Number of Training Epochs', type=int, default=200)
    # Active Learning Args
    parser.add_argument('--alg', help='acquisition algorithm, rand, conf, marg, badge, coreset', type=str, default='rand')
    parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=500)
    parser.add_argument('--nStart', help='number of points to start', type=int, default=5000)
    parser.add_argument('--nEnd', help='total number of points to query', type=int, default=15000)
    # Method Args
    parser.add_argument('--method', help='which method to use: [none, meta, prune]', type=str, default='none')
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def update_meter(train_avg_acc, train_minority_acc, train_majority_acc, logits, y, p):
    preds = torch.argmax(logits, axis=1)
    correct_batch = (preds == y)
    train_avg_acc.update(correct_batch.sum().item() / len(y), len(y))
    mask = y != p
    n = mask.sum().item()
    if n != 0:
        corr = correct_batch[mask].sum().item()
        train_minority_acc.update(corr / n, n)

    # Update majority
    mask = y == p
    n = mask.sum().item()
    if n != 0:
        corr = correct_batch[mask].sum().item()
        train_majority_acc.update(corr / n, n)
    return train_avg_acc, train_minority_acc, train_majority_acc

def update_indicator(indicator, idxs, logits, y):
    preds = torch.argmax(logits, dim=1)
    correct_batch = (preds == y)
    indicator[idxs] = correct_batch.cpu()
    return indicator

def load_model(pretrained, architecture, num_classes):
    os.environ['TORCH_HOME'] = 'models/resnet'  # setting the environment variable
    if architecture == "resnet18":
        if pretrained:
            model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.resnet18(weights=None)
    else: # resnet50
        if pretrained:
            model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.resnet50(weights=None)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, num_classes)
    return model

def get_output(m, x):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)
    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)
    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    p = m.fc(x)
    return p, x
