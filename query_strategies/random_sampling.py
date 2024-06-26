import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, X, Y, labelled_mask, handler, args):
        super(RandomSampling, self).__init__(X, Y, labelled_mask, handler, args)

    def query(self, n):
        inds = np.where(self.labelled_mask==0)[0]
        return inds[np.random.permutation(len(inds))][:n]
