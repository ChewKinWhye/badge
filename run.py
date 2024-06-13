from query_strategies import RandomSampling, BadgeSampling, LeastConfidence, MarginSampling, EntropySampling, CoreSet
from utils.dataset import get_dataset, get_handler
from utils.utils import parse_args

import numpy as np
import torch
import gc
import os

if __name__ == "__main__":
    # Parse args
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset as Numpy arrays
    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = get_dataset(args.dataset, args.path)
    # Load data handler
    handler = get_handler(args.data)

    # Initial labelled pool, Randomly select nStart indices to label
    labelled_mask = np.zeros(len(X_tr), dtype=bool)
    labelled_mask[np.random.choice(len(X_tr), args.nStart, replace=False)] = True

    # Acquisition Algorithm
    if args.alg == 'rand': # random sampling
        strategy = RandomSampling(X_tr, Y_tr, labelled_mask, handler, args)
    elif args.alg == 'conf': # confidence-based sampling
        strategy = LeastConfidence(X_tr, Y_tr, labelled_mask, handler, args)
    elif args.alg == 'marg': # margin-based sampling
        strategy = MarginSampling(X_tr, Y_tr, labelled_mask, handler, args)
    elif args.alg == 'badge': # batch active learning by diverse gradient embeddings
        strategy = BadgeSampling(X_tr, Y_tr, labelled_mask, handler, args)
    elif args.alg == 'coreset': # coreset sampling
        strategy = CoreSet(X_tr, Y_tr, labelled_mask, handler, args)
    elif args.alg == 'entropy': # entropy-based sampling
        strategy = EntropySampling(X_tr, Y_tr, labelled_mask, handler, args)
    else:
        print('Choose a valid acquisition function.')
        raise ValueError

    # Stats
    NUM_ROUNDS = (args.nEnd - args.nStart) // args.nQuery
    acc = np.zeros(NUM_ROUNDS+1)

    # Round 0 Train and Test
    strategy.train(X_val, Y_val, verbose=True) # Print out first round of training
    acc[0] = strategy.evaluate(X_te, Y_te)
    print(f"Round 0, Train Size: {np.sum(labelled_mask)}: Test accuracy: {acc[0]}")

    for rd in range(1, NUM_ROUNDS+1):
        # Query
        query_idxs = strategy.query(args.nQuery)
        labelled_mask[query_idxs] = True
        strategy.update(labelled_mask)

        # Round Train and Test
        strategy.train(X_val, Y_val, verbose=False)
        acc[rd] = strategy.evaluate(X_te, Y_te)

        # Print and Clean up
        print(f"Round {rd}, Train Size: {np.sum(labelled_mask)}: Test accuracy: {acc[rd]}")
        torch.cuda.empty_cache()
        gc.collect()
