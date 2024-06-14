from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.model import get_model

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os

class Strategy:
    def __init__(self, X, Y, labelled_mask, handler, args):
        self.X = X
        self.Y = Y
        self.labelled_mask = labelled_mask
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.clf = get_model(args.model)
    def query(self, n):
        pass

    def update(self, labelled_mask):
        self.labelled_mask = labelled_mask
 
    def train(self, X_val, Y_val, verbose=True):
        # Initialize model and optimizer
        self.clf = get_model(self.args.model).cuda()
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Obtain train and validation dataset and loader
        idxs_train = np.arange(self.n_pool)[self.labelled_mask]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y[idxs_train]).long(), is_train=True),
                               shuffle=True, batch_size=self.args.batch_size)

        best_epoch, best_val_acc = 0, 0
        for epoch in range(self.args.num_epochs):
            # Train
            train_acc, train_loss = 0, 0
            self.clf.train()
            for x, y, idxs in loader_tr:
                # Forward Pass
                x, y = Variable(x.cuda()), Variable(y.cuda())
                optimizer.zero_grad()
                out, _ = self.clf(x)
                loss = F.cross_entropy(out, y)
                # Backward Pass
                loss.backward()
                # Update
                for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
                optimizer.step()
                # Logging
                train_acc += torch.sum((torch.max(out, 1)[1] == y).float()).data.item() / len(loader_tr.dataset)
                train_loss += loss.detach().item() / len(loader_tr)
            self.clf.eval()
            validation_acc = self.evaluate(X_val, Y_val)
            if validation_acc > best_val_acc:
                best_val_acc = validation_acc
                best_epoch = epoch
                torch.save(self.clf.state_dict(), os.path.join(self.args.save_dir, "best_model.pth"))
            # Print
            if verbose:
                print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Validation Acc: {validation_acc:.4f}")
        if verbose:
            print(f"Loading model with highest validation accuracy of {best_val_acc:.4f} at epoch {best_epoch}")
        self.clf.load_state_dict(torch.load(os.path.join(self.args.save_dir, "best_model.pth")))


    def retrain(self, X_query, Y_query, X_val, Y_val):
        # Extract Embeddings
        query_emb = self.get_embedding(X_query, Y_query)
        val_emb = self.get_embedding(X_val, Y_val)

        # Select best c value to be used for logistic regression
        C_OPTIONS = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 20, 50]
        val_accuracies = []
        for c in C_OPTIONS:
            cls_head = LogisticRegression(penalty="l1", C=c, solver="liblinear")
            cls_head.fit(query_emb, Y_query)
            preds_val = cls_head.predict(val_emb)
            val_accuracies.append((preds_val == Y_val).sum().item() / len(Y_val))
        best_c_value = C_OPTIONS[np.argmax(val_accuracies)]
        print(f"Validation Accuracies: {val_accuracies}")
        print(f"Best C value chosen: {best_c_value}")

        # Train logistic regression with best c value found
        cls_head = LogisticRegression(penalty="l1", C=best_c_value, solver="liblinear")
        cls_head.fit(query_emb, Y_query)

        # Update model with retrained classification head
        with torch.no_grad():
            self.clf.linear.weight.copy_(torch.from_numpy(cls_head.coef_).float())
            self.clf.linear.bias.copy_(torch.from_numpy(cls_head.intercept_).float())
        self.clf.cuda()

    def evaluate(self, X, Y):
        Y = torch.Tensor(Y)
        P = self.predict(X, Y)
        accuracy = 1.0 * (Y == P).sum().item() / len(Y)
        return accuracy

    def predict(self, X, Y):
        self.clf.eval()
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), is_train=False),
                               shuffle=False, batch_size=self.args.batch_size)

        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in data_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P


    def predict_prob(self, X, Y):
        self.clf.eval()
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), is_train=False),
                               shuffle=False, batch_size=self.args.batch_size)
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in data_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                out = F.softmax(out, dim=1)
                probs[idxs] = out.cpu().data
        
        return probs


    def get_embedding(self, X, Y, return_probs=False):
        self.clf.eval()
        data_loader = DataLoader(self.handler(X, torch.Tensor(Y).long(), is_train=False),
                               shuffle=False, batch_size=self.args.batch_size)

        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        probs = torch.zeros(len(Y), self.clf.linear.out_features)
        with torch.no_grad():
            for x, y, idxs in data_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()
                if return_probs:
                     pr = F.softmax(out,1)
                     probs[idxs] = pr.data.cpu()
        if return_probs: return embedding, probs
        return embedding
