from sklearn.metrics import pairwise_distances
from .strategy import Strategy
import numpy as np

class CoreSet(Strategy):
    def __init__(self, X, Y, P, labelled_mask, handler, num_classes, num_attributes, num_epochs, target_resolution, test_group, args, tor=1e-4):
        super(CoreSet, self).__init__(X, Y, P, labelled_mask, handler, num_classes, num_attributes, num_epochs, target_resolution, test_group, args)
        self.tor = tor

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.labelled_mask]
        lb_flag = self.labelled_mask.copy()
        _, embedding = self.predict_output(self.X, self.Y)
        embedding = embedding.numpy()

        chosen = self.furthest_first(embedding[idxs_unlabeled, :], embedding[lb_flag, :], n)

        return idxs_unlabeled[chosen]
