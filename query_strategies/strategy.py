from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.utils import AverageMeter, update_meter, load_model, get_output
import time
import tqdm

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os

class Strategy:
    def __init__(self, X, Y, P, labelled_mask, handler, num_classes, num_epochs, args):
        self.X = X
        self.Y = Y
        self.P = P
        self.labelled_mask = labelled_mask
        self.handler = handler
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.args = args
        self.n_pool = len(Y)

    def query(self, n):
        pass

    def update(self, labelled_mask):
        self.labelled_mask = labelled_mask

    def train(self, X_val, Y_val, P_val, verbose=True):
        # Initialize model and optimizer
        self.clf = load_model(self.args.pretrained, self.args.architecture, self.num_classes).cuda()
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_train = np.arange(self.n_pool)[self.labelled_mask].astype(int)
        loader_tr = DataLoader(self.handler([self.X[i] for i in idxs_train], torch.Tensor(self.Y[idxs_train]).long(), torch.Tensor(self.P[idxs_train]).long(), isTrain=True),
                               shuffle=True, batch_size=self.args.batch_size)
        loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False),
                               shuffle=False, batch_size=self.args.batch_size)

        criterion = torch.nn.CrossEntropyLoss()

        # --- Train Start ---
        best_val_avg_acc, best_epoch = -1, None
        for epoch in range(self.num_epochs):
            self.clf.train()
            # Track metrics
            ce_loss_meter, train_minority_acc, train_majority_acc, train_avg_acc = AverageMeter(), AverageMeter(), \
                                                                                   AverageMeter(), AverageMeter()
            start = time.time()
            for batch in tqdm.tqdm(loader_tr, disable=True):
                x, y, p, idxs = batch
                x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                optimizer.zero_grad()

                # Cross Entropy Loss
                logits = self.clf(x)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                # Monitor training stats
                ce_loss_meter.update(torch.mean(loss).detach().item(), x.size(0))
                train_avg_acc, train_minority_acc, train_majority_acc = update_meter(train_avg_acc, train_minority_acc,
                                                                                     train_majority_acc,
                                                                                     logits.detach(), y, p)

            self.clf.eval()
            val_minority_acc, val_majority_acc, val_avg_acc = self.evaluate_model(loader_val)
            # Save best model based on worst group accuracy
            if val_avg_acc > best_val_avg_acc:
                torch.save(self.clf.state_dict(), os.path.join(self.args.save_dir, "ckpt.pt"))
                best_val_avg_acc = val_avg_acc
                best_epoch = epoch
            # Print stats
            if verbose:
                print(f"Epoch {epoch} Loss: {ce_loss_meter.avg:.3f} Time Taken: {time.time() - start:.3f}")
                print(f"Train Average Accuracy: {train_avg_acc.avg:.3f} Train Majority Accuracy: {train_majority_acc.avg:.3f} "
                    f"Train Minority Accuracy: {train_minority_acc.avg:.3f}")
                print(f"Val Average Accuracy: {val_avg_acc:.3f} Val Majority Accuracy: {val_majority_acc:.3f} "
                      f"Val Minority Accuracy: {val_minority_acc:.3f}")
        # --- Train End ---
        print(f'Best validation accuracy: {best_val_avg_acc:.3f} at epoch {best_epoch}')
        state_dict = torch.load(os.path.join(self.args.save_dir, "ckpt.pt"))
        return state_dict

    def evaluate_model(self, loader):
        self.clf.eval()
        minority_acc, majority_acc, avg_acc = AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            for x, y, p, idxs in tqdm.tqdm(loader, disable=True):
                x, y, p = x.cuda(), y.cuda(), p.cuda()
                logits = self.clf(x)
                avg_acc, minority_acc, majority_acc = update_meter(avg_acc, minority_acc, majority_acc, logits, y, p)

        self.clf.train()
        return minority_acc.avg, majority_acc.avg, avg_acc.avg

    def predict(self, X, Y):
        self.clf.eval()
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), isTrain=False),
                               shuffle=False, batch_size=self.args.batch_size)

        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in data_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_output(self, X, Y):
        self.clf.eval()
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), isTrain=False),
                                 shuffle=False, batch_size=self.args.batch_size)
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        embedding = torch.zeros([len(Y), 512])

        with torch.no_grad():
            for x, y, idxs in data_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                p, emb = get_output(self.clf, x)
                p = F.softmax(p, dim=1)
                probs[idxs] = p.cpu().data
                embedding[idxs] = emb.data.cpu()
        return probs, embedding
