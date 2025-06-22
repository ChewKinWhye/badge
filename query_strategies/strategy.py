from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.autograd import Variable
from utils.utils import AverageMeter, get_output, AverageGroupMeter, infinite_dataloader
import time
import tqdm
from utils.model import get_model
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os
import copy
import higher

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
        self.labelled_mask = labelled_mask.astype(bool)

    def train(self, X_val, Y_val, P_val, verbose=True):
        # Initialize model and optimizer
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        if self.args.architecture == "BERT":
            optimizer = optim.AdamW(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_train = np.arange(self.n_pool)[self.labelled_mask].astype(int)

        loader_tr = DataLoader(self.handler([self.X[i] for i in idxs_train], torch.Tensor(self.Y[idxs_train]).long(), torch.Tensor(self.P[idxs_train]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True)
        loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False, target_resolution=self.target_resolution),
                               shuffle=False, batch_size=self.args.batch_size)

        criterion = torch.nn.CrossEntropyLoss()

        # --- Train Start ---
        best_val_min_acc, best_epoch = -1, None
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
                torch.nn.utils.clip_grad_norm_(self.clf.parameters(), 1.0)
                optimizer.step()

                # Monitor training stats
                ce_loss_meter.update(torch.mean(loss).detach().item(), x.size(0))
                group_acc.update(logits.detach(), y, p)
            train_avg_acc, train_minority_acc, train_majority_acc = group_acc.get_stats(self.test_group)

            self.clf.eval()
            val_avg_acc, val_minority_acc, val_majority_acc = self.evaluate_model(loader_val)
            # Save best model based on worst group accuracy
            if val_minority_acc > best_val_min_acc:
                torch.save(self.clf.state_dict(), os.path.join(self.args.save_dir, "ckpt.pt"))
                best_val_min_acc = val_minority_acc
                best_epoch = epoch
            # Print stats
            if verbose:
                print(f"Epoch {epoch} Loss: {ce_loss_meter.avg:.3f} Time Taken: {time.time() - start:.3f}")
                print(f"Train Average Accuracy: {train_avg_acc:.3f} Train Majority Accuracy: {train_majority_acc:.3f} "
                    f"Train Minority Accuracy: {train_minority_acc:.3f}")
                print(f"Val Average Accuracy: {val_avg_acc:.3f} Val Majority Accuracy: {val_majority_acc:.3f} "
                      f"Val Minority Accuracy: {val_minority_acc:.3f}")
        # --- Train End ---
        print(f'Best validation minority accuracy: {best_val_min_acc:.3f} at epoch {best_epoch}')
        state_dict = torch.load(os.path.join(self.args.save_dir, "ckpt.pt"))
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        self.clf.load_state_dict(state_dict)
        return state_dict


    def train_meta_reweight(self, labelled_mask, X_val, Y_val, P_val, verbose=True):
        # Initialize model and optimizer
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_metatrain = np.arange(self.n_pool)[(labelled_mask < np.max(labelled_mask)) & (labelled_mask != 0)].astype(int)
        idxs_metatest = np.arange(self.n_pool)[(labelled_mask == np.max(labelled_mask))].astype(int)

        loader_metatrain = DataLoader(self.handler([self.X[i] for i in idxs_metatrain], torch.Tensor(self.Y[idxs_metatrain]).long(), torch.Tensor(self.P[idxs_metatrain]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True)
        dataset_weights = torch.nn.Parameter(torch.ones(len(idxs_metatrain)).cuda(), requires_grad=True)
        weight_optimizer = optim.Adam([dataset_weights], lr=0.1)

        loader_metatest = infinite_dataloader(DataLoader(self.handler([self.X[i] for i in idxs_metatest], torch.Tensor(self.Y[idxs_metatest]).long(), torch.Tensor(self.P[idxs_metatest]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True))

        loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False, target_resolution=self.target_resolution),
                               shuffle=False, batch_size=self.args.batch_size)

        criterion = torch.nn.CrossEntropyLoss()

        # --- Train Start ---
        best_val_min_acc, best_epoch = -1, None

        for epoch in range(self.num_epochs):
            self.clf.train()
            # Track metrics
            ce_loss_meter, train_group_acc = AverageMeter(), AverageGroupMeter(self.num_classes, self.num_attributes)
            start = time.time()
            for batch in tqdm.tqdm(loader_metatrain, disable=True):
                x, y, p, idxs = batch
                x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                sample_weights = dataset_weights[idxs]
                sample_weights = sample_weights / torch.sum(sample_weights)
                weight_optimizer.zero_grad()
                optimizer.zero_grad()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.args.inner_lr
                self.clf.eval()
                with higher.innerloop_ctx(self.clf, optimizer, copy_initial_weights=False) as (fnet, diffopt):
                    for _ in range(self.args.inner_steps):
                        logits = fnet(x)
                        sample_loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                        loss = torch.sum(sample_loss * sample_weights)
                        diffopt.step(loss)

                    x_meta, y_meta, p_meta, idxs_meta = next(loader_metatest)
                    x_meta, y_meta, p_meta, idxs_meta = x_meta.cuda(), y_meta.cuda(), p_meta.cuda(), idxs_meta.cuda()
                    logits_meta = fnet(x_meta)
                    meta_loss = criterion(logits_meta, y_meta)
                    weight_optimizer.zero_grad()
                    meta_loss.backward()
                weight_optimizer.step()
                with torch.no_grad():
                    dataset_weights.clamp_(min=0.1)
                # Outer-loop Optimizations
                self.clf.train()
                optimizer.zero_grad()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.args.lr
                logits = self.clf(x)
                sample_loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                loss = torch.sum(sample_loss * sample_weights.detach())
                loss += criterion(self.clf(x_meta), y_meta)
                loss.backward()
                optimizer.step()

            # Meta Evaluation, evaluate after updating on train dataset
            self.clf.eval()

            train_avg_acc, train_minority_acc, train_majority_acc = train_group_acc.get_stats(self.test_group)
            val_avg_acc, val_minority_acc, val_majority_acc = self.evaluate_model(loader_val)

            # Save best model based on worst group accuracy
            if val_minority_acc > best_val_min_acc:
                torch.save(self.clf.state_dict(), os.path.join(self.args.save_dir, "ckpt.pt"))
                best_val_min_acc = val_minority_acc
                best_epoch = epoch
            # Print stats
            if verbose:
                print(f"Epoch {epoch} Loss: {ce_loss_meter.avg:.3f} Time Taken: {time.time() - start:.3f}")
                print(f"Train Average Accuracy: {train_avg_acc:.3f} Train Majority/Best Accuracy: {train_majority_acc:.3f} "
                    f"Train Minority/Worst Accuracy: {train_minority_acc:.3f}")
                print(f"Val Average Accuracy: {val_avg_acc:.3f} Val Majority/Best Accuracy: {val_majority_acc:.3f} "
                      f"Val Minority/Worst Accuracy: {val_minority_acc:.3f}")
                print(f"Average Minority Weight: {torch.mean(dataset_weights[self.Y[idxs_metatrain]!=self.P[idxs_metatrain]])}")
                print(f"Average Majority Weight: {torch.mean(dataset_weights[self.Y[idxs_metatrain]==self.P[idxs_metatrain]])}")

        # --- Train End ---
        print(f'Best validation accuracy: {best_val_min_acc:.3f} at epoch {best_epoch}')
        state_dict = torch.load(os.path.join(self.args.save_dir, "ckpt.pt"))
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        self.clf.load_state_dict(state_dict)
        return state_dict


    def train_reweight_ANIL(self, labelled_mask, X_val, Y_val, P_val, verbose=True):
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_metatrain = np.arange(self.n_pool)[(labelled_mask < np.max(labelled_mask)) & (labelled_mask != 0)].astype(int)
        idxs_metatest = np.arange(self.n_pool)[(labelled_mask == np.max(labelled_mask))].astype(int)

        loader_metatrain = DataLoader(self.handler([self.X[i] for i in idxs_metatrain], torch.Tensor(self.Y[idxs_metatrain]).long(), torch.Tensor(self.P[idxs_metatrain]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True)
        dataset_weights = torch.nn.Parameter(torch.ones(len(idxs_metatrain)).cuda(), requires_grad=True)
        weight_optimizer = optim.Adam([dataset_weights], lr=0.1)

        loader_metatest = infinite_dataloader(DataLoader(self.handler([self.X[i] for i in idxs_metatest], torch.Tensor(self.Y[idxs_metatest]).long(), torch.Tensor(self.P[idxs_metatest]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True))

        loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False, target_resolution=self.target_resolution),
                               shuffle=False, batch_size=self.args.batch_size)

        criterion = torch.nn.CrossEntropyLoss()

        # --- Train Start ---
        # We want to run this such that it is enough to learn the sample weights
        for epoch in range(self.num_epochs):
            # Do not want to update the BN statistics
            self.clf.eval()
            for batch in tqdm.tqdm(loader_metatrain, disable=True):
                x, y, p, idxs = batch
                x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                sample_weights = dataset_weights[idxs]
                sample_weights = sample_weights / torch.sum(sample_weights)
                weight_optimizer.zero_grad()
                optimizer.zero_grad()
                # Inner-loop Optimizations
                if self.args.architecture == "BERT":
                    inner_optimizer = torch.optim.SGD(self.clf.model.classifier.parameters(), lr=self.args.inner_lr)
                else:
                    inner_optimizer = torch.optim.SGD(self.clf.fc.parameters(), lr=self.args.inner_lr)
                self.clf.eval()

                with higher.innerloop_ctx(self.clf, inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):
                    for _ in range(self.args.inner_steps):
                        logits = fnet(x)
                        sample_loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                        loss = torch.sum(sample_loss * sample_weights)
                        diffopt.step(loss)

                    x_meta, y_meta, p_meta, idxs_meta = next(loader_metatest)
                    x_meta, y_meta, p_meta, idxs_meta = x_meta.cuda(), y_meta.cuda(), p_meta.cuda(), idxs_meta.cuda()
                    logits_meta = fnet(x_meta)
                    meta_loss = criterion(logits_meta, y_meta)
                    weight_optimizer.zero_grad()
                    meta_loss.backward()
                weight_optimizer.step()
                with torch.no_grad():
                    dataset_weights.clamp_(min=0.1)
            with torch.no_grad():
                dataset_weights.data = dataset_weights.data / dataset_weights.data.mean()
            # Print learned dataset weights
            print(f"Epoch {epoch}")
            print(f"Average Minority Weight: {torch.mean(dataset_weights[self.Y[idxs_metatrain]!=self.P[idxs_metatrain]])}")
            print(f"Average Majority Weight: {torch.mean(dataset_weights[self.Y[idxs_metatrain]==self.P[idxs_metatrain]])}")

        # Now we simply use these learned sample weights to train the model
        # Initialize model and optimizer
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        sample_weights = torch.tensor([dataset_weights.detach().cpu().tolist()] + [1]*len(loader_metatest.dataset))
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),  # or any number of samples per epoch
                                        replacement=True)

        loader_tr = DataLoader(self.handler([self.X[i] for i in idxs_metatrain]+[self.X[i] for i in idxs_metatest],
                                            torch.Tensor(np.concatenate([self.Y[idxs_metatrain], self.Y[idxs_metatest]], axis=0)).long(),
                                            torch.Tensor(np.concatenate([self.P[idxs_metatrain], self.P[idxs_metatest]], axis=0)).long(),
                                            isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, sampler=sampler)
        # --- Train Start ---
        best_val_min_acc, best_epoch = -1, None

        for epoch in range(self.num_epochs):
            self.clf.train()
            # Track metrics
            ce_loss_meter, train_group_acc = AverageMeter(), AverageGroupMeter(self.num_classes, self.num_attributes)
            start = time.time()
            for batch in tqdm.tqdm(loader_tr, disable=True):
                x, y, p, idxs = batch
                x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                optimizer.zero_grad()
                logits = self.clf(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.clf.parameters(), 1.0)
                optimizer.step()
                # Monitor training stats
                ce_loss_meter.update(torch.mean(loss).detach().item(), x.size(0))
                train_group_acc.update(logits.detach(), y, p)

            # Meta Evaluation, evaluate after updating on train dataset
            self.clf.eval()

            train_avg_acc, train_minority_acc, train_majority_acc = train_group_acc.get_stats(self.test_group)
            val_avg_acc, val_minority_acc, val_majority_acc = self.evaluate_model(loader_val)

            # Save best model based on worst group accuracy
            if val_minority_acc > best_val_min_acc:
                torch.save(self.clf.state_dict(), os.path.join(self.args.save_dir, "ckpt.pt"))
                best_val_min_acc = val_minority_acc
                best_epoch = epoch
            # Print stats
            if verbose:
                print(f"Epoch {epoch} Loss: {ce_loss_meter.avg:.3f} Time Taken: {time.time() - start:.3f}")
                print(f"Train Average Accuracy: {train_avg_acc:.3f} Train Majority/Best Accuracy: {train_majority_acc:.3f} "
                    f"Train Minority/Worst Accuracy: {train_minority_acc:.3f}")
                print(f"Val Average Accuracy: {val_avg_acc:.3f} Val Majority/Best Accuracy: {val_majority_acc:.3f} "
                      f"Val Minority/Worst Accuracy: {val_minority_acc:.3f}")

        # --- Train End ---
        print(f'Best validation accuracy: {best_val_min_acc:.3f} at epoch {best_epoch}')
        state_dict = torch.load(os.path.join(self.args.save_dir, "ckpt.pt"))
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        self.clf.load_state_dict(state_dict)
        return state_dict

    def train_maml(self, labelled_mask, X_val, Y_val, P_val, verbose=True):
        # Initialize model and optimizer
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        if self.args.architecture == "BERT":
            optimizer = optim.AdamW(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_metatrain = np.arange(self.n_pool)[(labelled_mask < np.max(labelled_mask)) & (labelled_mask != 0)].astype(int)
        idxs_metatest = np.arange(self.n_pool)[(labelled_mask == np.max(labelled_mask))].astype(int)

        loader_metatrain = DataLoader(self.handler([self.X[i] for i in idxs_metatrain], torch.Tensor(self.Y[idxs_metatrain]).long(), torch.Tensor(self.P[idxs_metatrain]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True)

        loader_metatest = infinite_dataloader(DataLoader(self.handler([self.X[i] for i in idxs_metatest], torch.Tensor(self.Y[idxs_metatest]).long(), torch.Tensor(self.P[idxs_metatest]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True))

        loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False, target_resolution=self.target_resolution),
                               shuffle=False, batch_size=self.args.batch_size)

        criterion = torch.nn.CrossEntropyLoss()

        # --- Train Start ---
        best_val_min_acc, best_epoch = -1, None

        for epoch in range(self.num_epochs):
            self.clf.train()
            # Track metrics
            ce_loss_meter, train_group_acc = AverageMeter(), AverageGroupMeter(self.num_classes, self.num_attributes)
            start = time.time()
            for batch in tqdm.tqdm(loader_metatrain, disable=True):
                x, y, p, idxs = batch
                x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()

                # Inner-loop Optimizations
                optimizer.zero_grad()
                self.clf.eval()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.args.inner_lr
                with higher.innerloop_ctx(self.clf, optimizer, copy_initial_weights=False) as (fnet, diffopt):
                    for _ in range(self.args.inner_steps):
                        logits = fnet(x)
                        loss = criterion(logits, y)
                        diffopt.step(loss)

                    x_meta, y_meta, p_meta, idxs_meta = next(loader_metatest)
                    x_meta, y_meta, p_meta, idxs_meta = x_meta.cuda(), y_meta.cuda(), p_meta.cuda(), idxs_meta.cuda()
                    logits_meta = fnet(x_meta)
                    meta_loss = criterion(logits_meta, y_meta)
                    optimizer.zero_grad()
                    meta_loss.backward()

                # Outer-loop Optimizations
                self.clf.train()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.args.lr
                logits = self.clf(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            # Meta Evaluation, evaluate after updating on train dataset
            self.clf.eval()

            train_avg_acc, train_minority_acc, train_majority_acc = train_group_acc.get_stats(self.test_group)
            val_avg_acc, val_minority_acc, val_majority_acc = self.evaluate_model(loader_val)

            # Save best model based on worst group accuracy
            if val_minority_acc > best_val_min_acc:
                torch.save(self.clf.state_dict(), os.path.join(self.args.save_dir, "ckpt.pt"))
                best_val_min_acc = val_minority_acc
                best_epoch = epoch
            # Print stats
            if verbose:
                print(f"Epoch {epoch} Loss: {ce_loss_meter.avg:.3f} Time Taken: {time.time() - start:.3f}")
                print(f"Train Average Accuracy: {train_avg_acc:.3f} Train Majority/Best Accuracy: {train_majority_acc:.3f} "
                    f"Train Minority/Worst Accuracy: {train_minority_acc:.3f}")
                print(f"Val Average Accuracy: {val_avg_acc:.3f} Val Majority/Best Accuracy: {val_majority_acc:.3f} "
                      f"Val Minority/Worst Accuracy: {val_minority_acc:.3f}")

        # --- Train End ---
        print(f'Best validation accuracy: {best_val_min_acc:.3f} at epoch {best_epoch}')
        state_dict = torch.load(os.path.join(self.args.save_dir, "ckpt.pt"))
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        self.clf.load_state_dict(state_dict)
        return state_dict

    def train_fomaml(self, labelled_mask, X_val, Y_val, P_val, verbose=True):
        # Initialize model and optimizer
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        if self.args.architecture == "BERT":
            optimizer = optim.AdamW(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_metatrain = np.arange(self.n_pool)[(labelled_mask < np.max(labelled_mask)) & (labelled_mask != 0)].astype(int)
        idxs_metatest = np.arange(self.n_pool)[(labelled_mask == np.max(labelled_mask))].astype(int)

        loader_metatrain = DataLoader(self.handler([self.X[i] for i in idxs_metatrain], torch.Tensor(self.Y[idxs_metatrain]).long(), torch.Tensor(self.P[idxs_metatrain]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True)

        loader_metatest = infinite_dataloader(DataLoader(self.handler([self.X[i] for i in idxs_metatest], torch.Tensor(self.Y[idxs_metatest]).long(), torch.Tensor(self.P[idxs_metatest]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True))

        loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False, target_resolution=self.target_resolution),
                               shuffle=False, batch_size=self.args.batch_size)

        criterion = torch.nn.CrossEntropyLoss()

        # --- Train Start ---
        best_val_min_acc, best_epoch = -1, None

        for epoch in range(self.num_epochs):
            self.clf.train()
            # Track metrics
            ce_loss_meter, train_group_acc = AverageMeter(), AverageGroupMeter(self.num_classes, self.num_attributes)
            start = time.time()
            for batch in tqdm.tqdm(loader_metatrain, disable=True):
                x, y, p, idxs = batch
                x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
                # Save initial model
                optimizer.zero_grad()
                self.clf.eval()
                state_dict_copy = copy.deepcopy(self.clf.state_dict())
                optimizer_state_copy = copy.deepcopy(optimizer.state_dict())

                # Inner-loop optimization
                for _ in range(self.args.inner_steps):
                    optimizer.zero_grad()
                    logits = self.clf(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                self.clf.train()
                # Obtain Meta-Gradients
                optimizer.zero_grad()
                x_meta, y_meta, p_meta, idxs_meta = next(loader_metatest)
                x_meta, y_meta, p_meta, idxs_meta = x_meta.cuda(), y_meta.cuda(), p_meta.cuda(), idxs_meta.cuda()
                logits_meta = self.clf(x_meta)
                meta_loss = criterion(logits_meta, y_meta)
                meta_grads = torch.autograd.grad(meta_loss, self.clf.parameters())

                optimizer.zero_grad()
                self.clf.load_state_dict(state_dict_copy)
                optimizer.load_state_dict(optimizer_state_copy)

                # Outer-loop optimization
                for meta_param, grad in zip(self.clf.parameters(), meta_grads):
                    meta_param.grad = grad.clone()
                logits = self.clf(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            # Meta Evaluation, evaluate after updating on train dataset
            self.clf.eval()

            train_avg_acc, train_minority_acc, train_majority_acc = train_group_acc.get_stats(self.test_group)
            val_avg_acc, val_minority_acc, val_majority_acc = self.evaluate_model(loader_val)

            # Save best model based on worst group accuracy
            if val_minority_acc > best_val_min_acc:
                torch.save(self.clf.state_dict(), os.path.join(self.args.save_dir, "ckpt.pt"))
                best_val_min_acc = val_minority_acc
                best_epoch = epoch
            # Print stats
            if verbose:
                print(f"Epoch {epoch} Loss: {ce_loss_meter.avg:.3f} Time Taken: {time.time() - start:.3f}")
                print(f"Train Average Accuracy: {train_avg_acc:.3f} Train Majority/Best Accuracy: {train_majority_acc:.3f} "
                    f"Train Minority/Worst Accuracy: {train_minority_acc:.3f}")
                print(f"Val Average Accuracy: {val_avg_acc:.3f} Val Majority/Best Accuracy: {val_majority_acc:.3f} "
                      f"Val Minority/Worst Accuracy: {val_minority_acc:.3f}")

        # --- Train End ---
        print(f'Best validation accuracy: {best_val_min_acc:.3f} at epoch {best_epoch}')
        state_dict = torch.load(os.path.join(self.args.save_dir, "ckpt.pt"))
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        self.clf.load_state_dict(state_dict)
        return state_dict


    def train_ANIL(self, labelled_mask, X_val, Y_val, P_val, verbose=True):
        # Initialize model and optimizer
        self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes)
        self.clf = self.clf.cuda()
        if self.args.architecture == "BERT":
            optimizer = optim.AdamW(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_metatrain = np.arange(self.n_pool)[(labelled_mask < np.max(labelled_mask)) & (labelled_mask != 0)].astype(int)
        idxs_metatest = np.arange(self.n_pool)[(labelled_mask == np.max(labelled_mask))].astype(int)

        loader_metatrain = DataLoader(self.handler([self.X[i] for i in idxs_metatrain], torch.Tensor(self.Y[idxs_metatrain]).long(), torch.Tensor(self.P[idxs_metatrain]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True)

        loader_metatest = infinite_dataloader(DataLoader(self.handler([self.X[i] for i in idxs_metatest], torch.Tensor(self.Y[idxs_metatest]).long(), torch.Tensor(self.P[idxs_metatest]).long(), isTrain=True, target_resolution=self.target_resolution),
                               batch_size=self.args.batch_size, shuffle=True))

        loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False, target_resolution=self.target_resolution),
                               shuffle=False, batch_size=self.args.batch_size)

        criterion = torch.nn.CrossEntropyLoss()

        # --- Train Start ---
        best_val_min_acc, best_epoch = -1, None

        for epoch in range(self.num_epochs):
            self.clf.train()
            # Track metrics
            ce_loss_meter, train_group_acc = AverageMeter(), AverageGroupMeter(self.num_classes, self.num_attributes)
            start = time.time()
            for batch in tqdm.tqdm(loader_metatrain, disable=True):
                x, y, p, idxs = batch
                x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()

                # Inner-loop Optimizations
                optimizer.zero_grad()
                if self.args.architecture == "BERT":
                    inner_optimizer = torch.optim.SGD(self.clf.model.classifier.parameters(), lr=self.args.inner_lr)
                else:
                    inner_optimizer = torch.optim.SGD(self.clf.fc.parameters(), lr=self.args.inner_lr)
                self.clf.eval()

                with higher.innerloop_ctx(self.clf, inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):
                    for _ in range(self.args.inner_steps):
                        logits = fnet(x)
                        loss = criterion(logits, y)
                        diffopt.step(loss)

                    x_meta, y_meta, p_meta, idxs_meta = next(loader_metatest)
                    x_meta, y_meta, p_meta, idxs_meta = x_meta.cuda(), y_meta.cuda(), p_meta.cuda(), idxs_meta.cuda()
                    logits_meta = fnet(x_meta)
                    meta_loss = criterion(logits_meta, y_meta)
                    optimizer.zero_grad()
                    meta_loss.backward()
                self.clf.train()
                # Outer-loop Optimizations
                logits = self.clf(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            # Meta Evaluation, evaluate after updating on train dataset
            self.clf.eval()

            train_avg_acc, train_minority_acc, train_majority_acc = train_group_acc.get_stats(self.test_group)
            val_avg_acc, val_minority_acc, val_majority_acc = self.evaluate_model(loader_val)

            # Save best model based on worst group accuracy
            if val_minority_acc > best_val_min_acc:
                torch.save(self.clf.state_dict(), os.path.join(self.args.save_dir, "ckpt.pt"))
                best_val_min_acc = val_minority_acc
                best_epoch = epoch
            # Print stats
            if verbose:
                print(f"Epoch {epoch} Loss: {ce_loss_meter.avg:.3f} Time Taken: {time.time() - start:.3f}")
                print(f"Train Average Accuracy: {train_avg_acc:.3f} Train Majority/Best Accuracy: {train_majority_acc:.3f} "
                    f"Train Minority/Worst Accuracy: {train_minority_acc:.3f}")
                print(f"Val Average Accuracy: {val_avg_acc:.3f} Val Majority/Best Accuracy: {val_majority_acc:.3f} "
                      f"Val Minority/Worst Accuracy: {val_minority_acc:.3f}")

        # --- Train End ---
        print(f'Best validation accuracy: {best_val_min_acc:.3f} at epoch {best_epoch}')
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
