from .strategy import Strategy
import gc
import torch
import numpy as np

def select(X, K, fisher, iterates, lamb=1, nLabeled=0):

    numEmbs = len(X)
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K))
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.cuda()

    # forward selection, over-sample by 2x
    print('forward selection...', flush=True)
    over_sample = 2
    for i in range(int(over_sample *  K)):

        # check trace with low-rank updates (woodbury identity)
        xt_ = X.cuda() 
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)

        # clear out gpu memory
        xt = xt_.cpu()
        del xt, innerInv
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # get the smallest unselected item
        traceEst = traceEst.detach().cpu().numpy()
        for j in np.argsort(traceEst)[::-1]:
            if j not in indsAll:
                ind = j
                break

        indsAll.append(ind)
        print(i, ind, traceEst[ind], flush=True)
       
        # commit to a low-rank update
        xt_ = X[ind].unsqueeze(0).cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

    # backward pruning
    print('backward pruning...', flush=True)
    for i in range(len(indsAll) - K):

        # select index for removal
        xt_ = X[indsAll].cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)


        # low-rank update (woodbury identity)
        xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]

    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    return indsAll

class BaitSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BaitSampling, self).__init__(X, Y, idxs_lb, net, handler, args)        
        self.lamb = args['lamb']

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # get low-rank point-wise fishers
        xt = self.get_exp_grad_embedding(self.X, self.Y)

        # get fisher
        print('getting fisher matrix...', flush=True)
        batchSize = 1000 # should be as large as gpu memory allows
        nClass = torch.max(self.Y).item() + 1
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        rounds = int(np.ceil(len(self.X) / batchSize))
        for i in range(int(np.ceil(len(self.X) / batchSize))):
            xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
            op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0).detach().cpu()
            fisher = fisher + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        # get fisher only for samples that have been seen before
        nClass = torch.max(self.Y).item() + 1
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[self.idxs_lb]
        rounds = int(np.ceil(len(xt2) / batchSize))
        for i in range(int(np.ceil(len(xt2) / batchSize))):
            xt_ = xt2[i * batchSize : (i + 1) * batchSize].cuda()
            op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt2)), 0).detach().cpu()
            init = init + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        chosen = select(xt[idxs_unlabeled], n, fisher, init, lamb=self.lamb, nLabeled=np.sum(self.idxs_lb))
        return idxs_unlabeled[chosen]
