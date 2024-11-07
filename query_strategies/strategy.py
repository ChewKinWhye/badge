from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.utils import AverageMeter, get_output, AverageGroupMeter
import time
import tqdm
from utils.model import get_model
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os
import learn2learn as l2l

class Strategy:
    def __init__(self, X, Y, P, labelled_mask, handler, num_classes, num_attributes, num_epochs, target_resolution, test_group, args):
        self.X = X
        self.Y = Y
        self.P = P
        self.labelled_mask = labelled_mask
        self.handler = handler
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.num_epochs = num_epochs
        self.target_resolution = target_resolution
        self.test_group = test_group
        self.args = args
        self.n_pool = len(Y)
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()

    def query(self, n):
        pass

    def update(self, labelled_mask):
        self.labelled_mask = labelled_mask

    def train(self, X_val, Y_val, P_val, state_dict=None, verbose=True):
        # Initialize model and optimizer
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        if state_dict is not None:
            self.clf.load_state_dict(state_dict)
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_train = np.arange(self.n_pool)[self.labelled_mask].astype(int)
        loader_tr = DataLoader(self.handler([self.X[i] for i in idxs_train], torch.Tensor(self.Y[idxs_train]).long(), torch.Tensor(self.P[idxs_train]).long(), isTrain=True, target_resolution=self.target_resolution),
                               shuffle=True, batch_size=self.args.batch_size)
        loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False, target_resolution=self.target_resolution),
                               shuffle=False, batch_size=self.args.batch_size)

        criterion = torch.nn.CrossEntropyLoss()

        # --- Train Start ---
        best_val_avg_acc, best_epoch = -1, None
        for epoch in range(self.num_epochs):
            self.clf.train()
            # Track metrics
            ce_loss_meter, group_acc = AverageMeter(), AverageGroupMeter(self.num_classes, self.num_attributes)
            start = time.time()
            for batch in tqdm.tqdm(loader_tr, disable=True):
                x, y, p, idxs = batch
                x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                optimizer.zero_grad()

                # Cross Entropy Loss
                logits = self.clf(x)
                loss = criterion(logits, y)

                loss.backward()
                if self.args.model == "BERT":
                    torch.nn.utils.clip_grad_norm_(self.clf.parameters(), 1.0)
                optimizer.step()

                # Monitor training stats
                ce_loss_meter.update(torch.mean(loss).detach().item(), x.size(0))
                group_acc.update(logits.detach(), y, p)
            train_avg_acc, train_minority_acc, train_majority_acc = group_acc.get_stats(self.test_group)

            self.clf.eval()
            val_avg_acc, val_minority_acc, val_majority_acc = self.evaluate_model(loader_val)
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
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        self.clf.load_state_dict(state_dict)
        return state_dict


    def train_MAML(self, X_query, Y_query, P_query, X_val, Y_val, P_val, verbose=True):
        # Initialize model and optimizer
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        maml = l2l.algorithms.MAML(self.clf, lr=self.args.lr, first_order=bool(self.args.first_order))
        optimizer = optim.Adam(maml.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_train = np.arange(self.n_pool)[self.labelled_mask].astype(int)
        loader_tr = DataLoader(self.handler([self.X[i] for i in idxs_train], torch.Tensor(self.Y[idxs_train]).long(), torch.Tensor(self.P[idxs_train]).long(), isTrain=True, target_resolution=self.target_resolution),
                               shuffle=True, batch_size=self.args.batch_size)
        loader_tr_meta = DataLoader(self.handler(X_query, torch.Tensor(Y_query).long(), torch.Tensor(P_query).long(), isTrain=True, target_resolution=self.target_resolution),
                               shuffle=True, batch_size=self.args.batch_size)
        loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False, target_resolution=self.target_resolution),
                               shuffle=False, batch_size=self.args.batch_size)

        criterion = torch.nn.CrossEntropyLoss()

        # --- Train Start ---
        best_val_avg_acc, best_epoch = -1, None
        for epoch in range(self.num_epochs):
            maml.module.train()
            # Track metrics
            ce_loss_meter, train_group_acc, val_group_acc = AverageMeter(), AverageGroupMeter(self.num_classes, self.num_attributes), AverageGroupMeter(self.num_classes, self.num_attributes)
            start = time.time()

            for batch in tqdm.tqdm(loader_tr_meta, disable=True):
                optimizer.zero_grad()
                # Copy weights
                task_model = maml.clone()  # torch.clone() for nn.Modules
                x_meta, y_meta, p_meta, idxs_meta = batch
                x_meta, y_meta, p_meta, idxs_meta = x_meta.cuda(), y_meta.cuda(), p_meta.cuda(), idxs_meta.cuda()
                # Take 5 update steps on the training dataset
                for k in range(self.args.inner_steps):
                    x, y, p, idxs = next(iter(loader_tr))
                    x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                    logits = task_model(x)
                    loss = criterion(logits, y)
                    task_model.adapt(loss)
                # Compute loss on query (meta) dataset
                logits_meta = task_model(x_meta)
                meta_loss = criterion(logits_meta, y_meta)
                meta_loss.backward()
                optimizer.step()

                # Monitor training stats
                ce_loss_meter.update(torch.mean(meta_loss).detach().item(), x_meta.size(0))
                train_group_acc.update(logits_meta.detach(), y_meta, p_meta)

            # Meta Evaluation, evaluate after updating on train dataset
            maml.module.eval()
            for batch in tqdm.tqdm(loader_val, disable=True):
                task_model = maml.clone()  # torch.clone() for nn.Modules
                x_meta, y_meta, p_meta, idxs_meta = batch
                x_meta, y_meta, p_meta, idxs_meta = x_meta.cuda(), y_meta.cuda(), p_meta.cuda(), idxs_meta.cuda()
                # Take 5 update steps on the training dataset
                for k in range(self.args.inner_steps):
                    x, y, p, idxs = next(iter(loader_tr))
                    x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                    logits = task_model(x)
                    loss = criterion(logits, y)
                    task_model.adapt(loss)
                # Compute loss on query (meta) dataset
                logits_meta = task_model(x_meta)

                # Monitor training stats
                val_group_acc.update(logits_meta.detach(), y_meta, p_meta)
            train_avg_acc, train_minority_acc, train_majority_acc = train_group_acc.get_stats(self.test_group)
            val_avg_acc, val_minority_acc, val_majority_acc = val_group_acc.get_stats(self.test_group)

            # Save best model based on worst group accuracy
            if val_avg_acc > best_val_avg_acc:
                torch.save(maml.module.state_dict(), os.path.join(self.args.save_dir, "ckpt.pt"))
                best_val_avg_acc = val_avg_acc
                best_epoch = epoch
            # Print stats
            if verbose:
                print(f"Epoch {epoch} Loss: {ce_loss_meter.avg:.3f} Time Taken: {time.time() - start:.3f}")
                print(f"Train Average Accuracy: {train_avg_acc:.3f} Train Majority/Best Accuracy: {train_majority_acc:.3f} "
                    f"Train Minority/Worst Accuracy: {train_minority_acc:.3f}")
                print(f"Val Average Accuracy: {val_avg_acc:.3f} Val Majority/Best Accuracy: {val_majority_acc:.3f} "
                      f"Val Minority/Worst Accuracy: {val_minority_acc:.3f}")

        # --- Train End ---
        print(f'Best validation accuracy: {best_val_avg_acc:.3f} at epoch {best_epoch}')
        state_dict = torch.load(os.path.join(self.args.save_dir, "ckpt.pt"))
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        self.clf.load_state_dict(state_dict)
        return state_dict

    def evaluate_model(self, loader, model=None):
        if model is None:
            model = self.clf
        model.eval()
        group_acc = AverageGroupMeter(self.num_classes, self.num_attributes)

        with torch.no_grad():
            for x, y, p, idxs in tqdm.tqdm(loader, disable=True):
                x, y, p = x.cuda(), y.cuda(), p.cuda()
                logits = model(x)
                group_acc.update(logits.detach(), y, p)
        avg_acc, minority_acc, majority_acc = group_acc.get_stats(self.test_group)
        model.train()
        return avg_acc, minority_acc, majority_acc

    def predict(self, X, Y):
        self.clf.eval()
        # Spurious Attribute does not matter
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), torch.Tensor(Y).long(), isTrain=False, target_resolution=self.target_resolution),
                               shuffle=False, batch_size=self.args.batch_size)

        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, _, idxs in data_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_output(self, X, Y):
        self.clf.eval()
        # Spurious Attribute does not matter
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), torch.Tensor(Y).long(), isTrain=False, target_resolution=self.target_resolution),
                                 shuffle=False, batch_size=self.args.batch_size)
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        if self.args.architecture == "resnet18" or self.args.architecture == "resnet50":
            embedding = torch.zeros([len(Y), 512])
        else:
            embedding = torch.zeros([len(Y), 768])
        with torch.no_grad():
            for x, y, _, idxs in data_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                p, emb = get_output(self.clf, x, self.args.architecture)
                p = F.softmax(p, dim=1)
                probs[idxs] = p.cpu().data
                embedding[idxs] = emb.data.cpu()
        return probs, embedding
