from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.utils import AverageMeter, update_meter, load_model, get_output
import time
import tqdm
from utils.model import get_model
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os
import copy
import torch.autograd as autograd

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
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes).cuda()
    def query(self, n):
        pass

    def update(self, labelled_mask):
        self.labelled_mask = labelled_mask

    def train(self, X_val, Y_val, P_val, verbose=True):
        # Initialize model and optimizer
        if self.args.method != "meta":
            self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes).cuda()
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
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes).cuda()
        self.clf.load_state_dict(state_dict, strict=False)
        return state_dict

    def train_prune(self, X_query, Y_query, P_query, X_val, Y_val, P_val, verbose=True):
        # Freeze all weight paramters since we are only updating the masks
        for name, param in self.clf.named_parameters():
            if "mask_weight" in name or "bn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, buff in self.clf.named_buffers():
            if "mask" in name:
                buff.fill_(True)

        optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader, dataset to train the mask is simply the queried points
        loader_tr = DataLoader(self.handler(X_query, torch.Tensor(Y_query).long(), torch.Tensor(P_query).long(), isTrain=True),
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
                logits = self.clf(x) # Set temperature as 1 allows for soft masking. For binary mask, temperature should increase with epoch
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
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes).cuda()
        self.clf.load_state_dict(state_dict, strict=False)
        return state_dict

    def train_MAML(self, X_query, Y_query, P_query, X_val, Y_val, P_val, verbose=True):
        # Initialize model and optimizer
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes).cuda()
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_train = np.arange(self.n_pool)[self.labelled_mask].astype(int)
        loader_tr = DataLoader(self.handler([self.X[i] for i in idxs_train], torch.Tensor(self.Y[idxs_train]).long(), torch.Tensor(self.P[idxs_train]).long(), isTrain=True),
                               shuffle=True, batch_size=self.args.batch_size)
        loader_tr_meta = DataLoader(self.handler(X_query, torch.Tensor(Y_query).long(), torch.Tensor(P_query).long(), isTrain=True),
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

            for batch in tqdm.tqdm(loader_tr_meta, disable=True):
                # Copy weights
                fast_weights = {name: param for name, param in self.clf.named_parameters() if param.requires_grad}
                x_meta, y_meta, p_meta, idxs_meta = batch
                x_meta, y_meta, p_meta, idxs_meta = x_meta.cuda(), y_meta.cuda(), p_meta.cuda(), idxs_meta.cuda()
                # Take 5 update steps on the training dataset
                for k in range(self.args.inner_steps):
                    x, y, p, idxs = next(iter(loader_tr))
                    x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                    logits = self.clf.meta_forward(x, fast_weights)
                    loss = criterion(logits, y)
                    if self.args.order == "first":
                        grad = torch.autograd.grad(loss, list(fast_weights.values()))
                    else:
                        grad = torch.autograd.grad(loss, list(fast_weights.values()), create_graph=True)
                    fast_weights = {name: param - self.args.lr * grad_part for (name, param), grad_part in zip(fast_weights.items(), grad)}
                # Compute loss on query (meta) dataset
                logits_meta = self.clf.meta_forward(x_meta, fast_weights)
                meta_loss = criterion(logits_meta, y_meta)
                optimizer.zero_grad()
                meta_loss.backward()
                optimizer.step()

                # Monitor training stats
                ce_loss_meter.update(torch.mean(meta_loss).detach().item(), x_meta.size(0))
                train_avg_acc, train_minority_acc, train_majority_acc = update_meter(train_avg_acc, train_minority_acc,
                                                                                     train_majority_acc,
                                                                                     logits_meta.detach(), y_meta, p_meta)

            # Meta Evaluation, evaluate after updating on train dataset
            self.clf.eval()
            clf_copy = copy.deepcopy(self.clf)
            for k in range(5):
                x, y, p, idxs = next(iter(loader_tr))
                x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                logits = clf_copy(x)
                loss = criterion(logits, y)
                grad = torch.autograd.grad(loss, clf_copy.parameters())
                with torch.no_grad():  # Ensure no gradients are tracked during this update
                    for param, grad in zip(clf_copy.parameters(), grad):
                        param -= self.args.lr * grad
            val_minority_acc, val_majority_acc, val_avg_acc = self.evaluate_model(loader_val, clf_copy)

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
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes).cuda()
        self.clf.load_state_dict(state_dict, strict=False)
        return state_dict

    def evaluate_model(self, loader, model=None):
        if model is None:
            model = self.clf
        model.eval()
        minority_acc, majority_acc, avg_acc = AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            for x, y, p, idxs in tqdm.tqdm(loader, disable=True):
                x, y, p = x.cuda(), y.cuda(), p.cuda()
                logits = model(x)
                avg_acc, minority_acc, majority_acc = update_meter(avg_acc, minority_acc, majority_acc, logits, y, p)

        model.train()
        return minority_acc.avg, majority_acc.avg, avg_acc.avg

    def predict(self, X, Y):
        self.clf.eval()
        # Spurious Attribute does not matter
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), torch.Tensor(Y).long(), isTrain=False),
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
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), torch.Tensor(Y).long(), isTrain=False),
                                 shuffle=False, batch_size=self.args.batch_size)
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        embedding = torch.zeros([len(Y), 512])

        with torch.no_grad():
            for x, y, _, idxs in data_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                p, emb = get_output(self.clf, x)
                p = F.softmax(p, dim=1)
                probs[idxs] = p.cpu().data
                embedding[idxs] = emb.data.cpu()
        return probs, embedding
