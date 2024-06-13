from .strategy import Strategy
import numpy as np

class MarginSampling(Strategy):
    def __init__(self, X, Y, labelled_mask, handler, args):
        super(MarginSampling, self).__init__(X, Y, labelled_mask, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.labelled_mask]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        return idxs_unlabeled[U.sort()[1].numpy()[:n]]
