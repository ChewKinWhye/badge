from .strategy import Strategy
import numpy as np

class MarginSampling(Strategy):
    def __init__(self, X, Y, P, labelled_mask, handler, num_classes, num_epochs, args):
        super(MarginSampling, self).__init__(X, Y, P, labelled_mask, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.labelled_mask]
        probs, _ = self.predict_output(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        return idxs_unlabeled[U.sort()[1].numpy()[:n]]
