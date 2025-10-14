import numpy as np
from .strategy import Strategy
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

class LossSampling(Strategy):
    def __init__(self, X, Y, P, labelled_mask, handler, num_classes, num_attributes, num_epochs, target_resolution, test_group, args):
        super(LossSampling, self).__init__(X, Y, P, labelled_mask, handler, num_classes, num_attributes, num_epochs, target_resolution, test_group, args)
        # For ResNet18
        self.h_dim = 128
        self.margin = 1
        self.layer_1 = torch.nn.Linear(64, self.h_dim)
        self.layer_2 = torch.nn.Linear(128, self.h_dim)
        self.layer_3 = torch.nn.Linear(256, self.h_dim)
        self.layer_4 = torch.nn.Linear(512, self.h_dim)
        self.layer_5 = torch.nn.Linear(self.h_dim*4, 1)
        self.relu = torch.nn.ReLU()

    def predict_loss(self, layer_1_out, layer_2_out, layer_3_out, layer_4_out):
        layer_1_features = self.relu(self.layer_1(torch.mean(layer_1_out, dim=(-1, -2)))) # (B, 64, H, W) -> (B, h_dim)
        layer_2_features = self.relu(self.layer_1(torch.mean(layer_2_out, dim=(-1, -2)))) # (B, 128, H, W) -> (B, h_dim)
        layer_3_features = self.relu(self.layer_1(torch.mean(layer_3_out, dim=(-1, -2)))) # (B, 256, H, W) -> (B, h_dim)
        layer_4_features = self.relu(self.layer_1(torch.mean(layer_4_out, dim=(-1, -2)))) # (B, 512, H, W) -> (B, h_dim)
        total_features = torch.cat((layer_1_features, layer_2_features, layer_3_features, layer_4_features), dim=-1) # (B, h_dim) -> (B, h_dim*4)
        predicted_loss = torch.flatten(self.relu(self.layer_5(total_features))) # (B, h_dim*4) -> (B)
        return predicted_loss

    def LPL_training_loss(self, predicted_loss, actual_loss):
        B = predicted_loss.size(0)
        num_comparisons = B * (B-1) / 2
        actual_pairwise_loss_difference = actual_loss[:, None] - actual_loss[None, :]
        actual_pairwise_loss_mask = actual_pairwise_loss_difference > 0
        predict_pairwise_loss_difference = predicted_loss[:, None] - predicted_loss[None, :]
        loss = self.relu(-actual_pairwise_loss_mask * predict_pairwise_loss_difference + self.margin) / num_comparisons
        return loss

    def predict_loss_dataset(self, X, Y):
        self.clf.eval()
        # Spurious Attribute does not matter
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), torch.Tensor(Y).long(), isTrain=False,
                                              target_resolution=self.target_resolution),
                                 shuffle=False, batch_size=self.args.batch_size)
        predicted_loss = torch.zeros(len(Y))
        with torch.no_grad():
            for x, y, _, idxs in data_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                e1_batch, e2_batch, e3_batch, e4_batch, _ = self.resnet_embedding_forward(x)
                predicted_loss_batch = self.predict_loss(e1_batch, e2_batch, e3_batch, e4_batch)
                predicted_loss[idxs] = predicted_loss_batch.data.cpu()
        return predicted_loss

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.labelled_mask]
        predicted_loss = self.predict_loss_dataset([self.X[i] for i in idxs_unlabeled], self.Y[idxs_unlabeled])
        return idxs_unlabeled[predicted_loss.sort()[1][-n:]]
