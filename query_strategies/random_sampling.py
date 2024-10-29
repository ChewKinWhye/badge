import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, X, Y, P, labelled_mask, handler, num_classes, num_epochs, target_resolution, args):
        super(RandomSampling, self).__init__(X, Y, P, labelled_mask, handler, num_classes, num_epochs, target_resolution, args)

    def query(self, n):
        inds = np.where(self.labelled_mask==0)[0]
        return inds[np.random.permutation(len(inds))][:n]
