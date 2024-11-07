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
    parser.add_argument('--architecture', help='model - resnet18, resnet50, ViT, BERT', type=str, default='resnet18')
    parser.add_argument("--pretrained", type=int, default=0, help="Use pretrained model")
    # Data Args
    parser.add_argument('--data_dir', help='data path', type=str, default='/hpctmp/e0200920')
    parser.add_argument('--dataset', help='dataset, mcdominoes, spawrious, spuco, treeperson', type=str, default='mcdominoes')
    parser.add_argument("--spurious_strength", type=float, default=0.95, help="Strength of spurious correlation")
    # Training Args
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--num_epochs', help='Number of Training Epochs', type=int, default=100)
    # Active Learning Args
    parser.add_argument('--alg', help='acquisition algorithm, rand, conf, marg, badge, coreset', type=str, default='rand')
    parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=500)
    parser.add_argument('--nStart', help='number of points to start', type=int, default=5000)
    parser.add_argument('--nEnd', help='total number of points to query', type=int, default=10000)
    # Method Args
    parser.add_argument('--method', help='which method to use: [none, meta]', type=str, default='none')
    # MAML Args
    parser.add_argument('--inner_steps', help='Number of inner steps for meta learning (usually 1-5)', type=int, default=5)
    parser.add_argument('--first_order', help='whether to use first-order approximation', type=int, default=0)

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

class AverageGroupMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, num_classes, num_attributes):
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.reset()

    def reset(self):
        self.avg = np.zeros((self.num_classes, self.num_attributes))
        self.sum = np.zeros((self.num_classes, self.num_attributes))
        self.count = np.zeros((self.num_classes, self.num_attributes))

    def update(self, logits, y, p):
        logits, y, p = logits.cpu(), y.cpu, p.cpu()
        # Add 1 to all the groups indexed by y and p to the count
        np.add.at(self.count, (y, p), 1)
        preds = torch.argmax(logits, axis=1)
        correct_batch = (preds == y)
        # Add 1 to all the groups indexed by y and p that are correct to the sum
        np.add.at(self.sum, (y[correct_batch], p[correct_batch]), 1)
        self.avg = self.sum / self.count

    def get_stats(self, test_group):
        average_acc = np.sum(self.sum) / np.sum(self.count)
        if test_group == "minority":
            minority_mask = ~np.eye(self.sum.shape[0], dtype=bool)
            minority_acc = np.sum(self.sum[minority_mask]) / np.sum(self.count[minority_mask])
            majority_mask = np.eye(self.sum.shape[0], dtype=bool)
            majority_acc = np.sum(self.sum[majority_mask]) / np.sum(self.count[majority_mask])
            return average_acc, minority_acc, majority_acc
        else:
            worst_acc = np.min(self.avg)
            best_acc = np.max(self.avg)
            return average_acc, worst_acc, best_acc

def get_output(m, x, model):
    if model == "resnet18" or model == "resnet50":
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
    else: #ViT
        x = m._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = m.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = m.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        p = m.heads(x)

        return p, x

# def update_meter(train_avg_acc, train_minority_acc, train_majority_acc, logits, y, p):
#     preds = torch.argmax(logits, axis=1)
#     correct_batch = (preds == y)
#     train_avg_acc.update(correct_batch.sum().item() / len(y), len(y))
#     mask = y != p
#     n = mask.sum().item()
#     if n != 0:
#         corr = correct_batch[mask].sum().item()
#         train_minority_acc.update(corr / n, n)
#
#     # Update majority
#     mask = y == p
#     n = mask.sum().item()
#     if n != 0:
#         corr = correct_batch[mask].sum().item()
#         train_majority_acc.update(corr / n, n)
#     return train_avg_acc, train_minority_acc, train_majority_acc
#
# def update_indicator(indicator, idxs, logits, y):
#     preds = torch.argmax(logits, dim=1)
#     correct_batch = (preds == y)
#     indicator[idxs] = correct_batch.cpu()
#     return indicator

# def load_model(pretrained, architecture, num_classes):
#     os.environ['TORCH_HOME'] = 'models/resnet'  # setting the environment variable
#     if architecture == "resnet18":
#         if pretrained:
#             model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         else:
#             model = torchvision.models.resnet18(weights=None)
#     else: # resnet50
#         if pretrained:
#             model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         else:
#             model = torchvision.models.resnet50(weights=None)
#     d = model.fc.in_features
#     model.fc = torch.nn.Linear(d, num_classes)
#     return model
