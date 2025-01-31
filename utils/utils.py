import numpy as np
import argparse
import random
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    # Model Args
    parser.add_argument('--architecture', help='model - resnet18, resnet50, BERT', type=str, default='resnet18')
    parser.add_argument("--pretrained", type=int, default=1, help="Use pretrained model")
    # Data Args
    parser.add_argument('--data_dir', help='data path', type=str, default='data')
    parser.add_argument('--dataset', help='dataset, mcdominoes, spuco, celeba, multinli, civilcomments', type=str, default='mcdominoes')
    parser.add_argument("--spurious_strength", type=float, default=0.95, help="Strength of spurious correlation, only tunable for some datasets")
    # Training Args
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=1e-2)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--num_epochs', help='Number of Training Epochs', type=int, default=100)
    # Active Learning Args
    parser.add_argument('--alg', help='acquisition algorithm, rand, conf, marg, badge, coreset', type=str, default='rand')
    parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
    parser.add_argument('--nStart', help='number of points to start', type=int, default=4500)
    parser.add_argument('--nEnd', help='total number of points to query', type=int, default=5000)
    # Method Args
    parser.add_argument('--method', help='which method to use: [none, mldgc]', type=str, default='none')

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        logits, y, p = logits.cpu(), y.cpu(), p.cpu()
        # Add 1 to all the groups indexed by y and p to the count
        np.add.at(self.count, (np.array(y, dtype=int), np.array(p, dtype=int)), 1)
        preds = torch.argmax(logits, axis=1)
        correct_batch = (preds == y)
        # Add 1 to all the groups indexed by y and p that are correct to the sum
        np.add.at(self.sum, (np.array(y[correct_batch], dtype=int), np.array(p[correct_batch], dtype=int)), 1)
        # Divide, setting avg to 0 when count is 0
        self.avg = np.divide(self.sum, self.count, out=np.zeros_like(self.sum, dtype=float), where=self.count != 0)

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

    else: # BERT
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = m.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        cls_token = last_hidden_state[:, 0]
        logits = outputs.logits
        return logits, cls_token

def infinite_dataloader(dataloader):
    """
    Creates an infinite generator for a PyTorch DataLoader.
    """
    while True:  # This ensures the function runs indefinitely
        for batch in dataloader:  # Iterates over the DataLoader
            yield batch  # Produces one batch at a time
