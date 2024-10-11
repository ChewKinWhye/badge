from .strategy import Strategy
import numpy as np
import torch

class EntropySampling(Strategy):
	def __init__(self, X, Y, P, labelled_mask, handler, num_classes, num_epochs, args):
		super(EntropySampling, self).__init__(X, Y, P, labelled_mask, handler, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.labelled_mask]
		probs, _ = self.predict_output(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		return idxs_unlabeled[U.sort()[1][:n]]
