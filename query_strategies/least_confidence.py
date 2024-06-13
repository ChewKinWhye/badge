from .strategy import Strategy
import numpy as np

class LeastConfidence(Strategy):
    def __init__(self, X, Y, labelled_mask, handler, args):
        super(LeastConfidence, self).__init__(X, Y, labelled_mask, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.labelled_mask]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
