from .strategy import Strategy
import numpy as np

class LeastConfidence(Strategy):
    def __init__(self, X, Y, P, labelled_mask, handler, num_classes, num_epochs, args):
        super(LeastConfidence, self).__init__(X, Y, P, labelled_mask, handler, num_classes, num_epochs, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.labelled_mask]
        probs, _ = self.predict_output([self.X[i] for i in idxs_unlabeled], self.Y[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
