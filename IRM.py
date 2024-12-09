# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

def torch_bernoulli(p, size):
    return (torch.rand(size) < p)


def color_mnist_images(images, numbers):
    # Normalize the images to [0, 1] if they are not already
    if images.max() > 1:
        images = images / 255.0

    # Define a fixed set of unique RGB colors for numbers 0-9
    color_map = {
        0: [255, 0, 0],     # Red
        1: [0, 255, 0],     # Green
        2: [0, 0, 255],     # Blue
        3: [255, 255, 0],   # Yellow
        4: [255, 0, 255],   # Magenta
        5: [0, 255, 255],   # Cyan
        6: [128, 128, 0],   # Olive
        7: [128, 0, 128],   # Purple
        8: [0, 128, 128],   # Teal
        9: [128, 128, 128], # Gray
    }

    # Convert the color map to a tensor
    color_map_tensor = torch.tensor([color_map[i] for i in range(10)], dtype=torch.float32)  # Shape: (10, 3)

    # Initialize an empty tensor to hold the colored images
    batch_size = images.shape[0]
    colored_images = torch.zeros((batch_size, 3, 28, 28), dtype=torch.float32)

    # Apply colors to each image based on the corresponding number
    for i in range(batch_size):
        rgb_color = color_map_tensor[numbers[i]]
        for channel in range(3):
            colored_images[i, channel, :, :] = images[i] * rgb_color[channel]

    return colored_images

def make_environment(images, labels, e):
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    minority_mask = torch_bernoulli(e, len(labels))
    colors = copy.deepcopy(labels)
    colors[minority_mask] = torch.randint(0, 10, (torch.sum(minority_mask),))
    # Assign a color based on the label; flip the color with probability e
    # Apply the color to the image by zeroing out the other color channel
    colored_images = color_mnist_images(images, colors)

    label_noise_mask = torch_bernoulli(0.25, len(labels))
    labels[label_noise_mask] = torch.randint(0, 10, (torch.sum(label_noise_mask),))
    return {
        'images': colored_images.cuda(),
        'labels': labels[:, None].cuda()
    }

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_train_accs = []
final_val_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)
    # Load MNIST, make train/val splits, and shuffle train set examples
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    mnist_test = datasets.MNIST('~/datasets/mnist', train=False, download=True)
    mnist_test = (mnist_test.data, mnist_test.targets)

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments
    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 1.0),
        make_environment(mnist_test[0], mnist_test[1], 1.0)
    ]


    # Define and instantiate the model

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(3 * 14 * 14, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, 10)
            for lin in [lin1, lin2, lin3]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

        def forward(self, input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 3, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], 3 * 14 * 14)
            out = self._main(out)
            return out


    mlp = MLP().cuda()

    criterion = torch.nn.CrossEntropyLoss()

    def mean_accuracy(logits, y):
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == y).sum().item()
        return correct / y.size(0) * 100


    def penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = criterion(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)


    # Train loop

    def pretty_print(*values):
        col_width = 13

        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)

        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))


    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'val acc', 'test acc')
    best_values = (0, 0, 0)
    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = criterion(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight
                          if step >= flags.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_acc = envs[2]['acc']
        test_acc = envs[3]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                val_acc.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy()
            )
        if val_acc.detach().cpu().numpy() > best_values[1]:
            best_values = (train_acc.detach().cpu().numpy(), val_acc.detach().cpu().numpy(), test_acc.detach().cpu().numpy())
    final_train_accs.append(best_values[0])
    final_val_accs.append(best_values[1])
    final_test_accs.append(best_values[2])
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final val acc (mean/std across restarts so far):')
    print(np.mean(final_val_accs), np.std(final_val_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))