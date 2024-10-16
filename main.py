from query_strategies import RandomSampling, BadgeSampling, LeastConfidence, MarginSampling, EntropySampling, CoreSet
from utils.dataset import get_data
from utils.utils import parse_args, set_seed
from torch.utils.data import DataLoader

import numpy as np
import torch
import gc
import os

if __name__ == "__main__":
    # Parse Args
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    # Load Dataset (Numpy Array)
    X_tr, Y_tr, P_tr, X_val, Y_val, P_val, X_te, Y_te, P_te, num_classes, _, handler = get_data(args.dataset, args.data_dir, args.spurious_strength,
                                               args.seed)

    # Initial labelled pool, Randomly select nStart indices to label
    labelled_mask = np.zeros(len(X_tr), dtype=bool)
    labelled_mask[np.random.choice(len(X_tr), args.nStart, replace=False)] = True

    # Acquisition Algorithm
    if args.alg == 'rand': # random sampling
        strategy = RandomSampling(X_tr, Y_tr, P_tr, labelled_mask, handler, num_classes, args.num_epochs, args)
    elif args.alg == 'conf': # confidence-based sampling
        strategy = LeastConfidence(X_tr, Y_tr, P_tr, labelled_mask, handler, num_classes, args.num_epochs, args)
    elif args.alg == 'badge': # batch active learning by diverse gradient embeddings
        strategy = BadgeSampling(X_tr, Y_tr, P_tr, labelled_mask, handler, num_classes, args.num_epochs, args)
    elif args.alg == 'coreset': # coreset sampling
        strategy = CoreSet(X_tr, Y_tr, P_tr, labelled_mask, handler, num_classes, args.num_epochs, args)
    else:
        print('Choose a valid acquisition function.')
        raise ValueError

    # Stats
    NUM_ROUNDS = (args.nEnd - args.nStart) // args.nQuery
    test_average_acc, test_minority_acc, test_majority_acc = np.zeros(NUM_ROUNDS+1), np.zeros(NUM_ROUNDS+1), np.zeros(NUM_ROUNDS+1)
    loader_test = DataLoader(handler(X_te, torch.Tensor(Y_te).long(), torch.Tensor(P_te).long(), isTrain=False),
                            shuffle=False, batch_size=args.batch_size)

    # Round 0 Train and Test
    state_dict = strategy.train(X_val, Y_val, P_val, verbose=True) # Print out first round of training
    test_minority_acc[0], test_majority_acc[0], test_average_acc[0] = strategy.evaluate_model(loader_test)
    print(f"Round 0, Train Size: {np.sum(labelled_mask)}, Test Average Accuracy: {test_average_acc[0]}, "
          f"Test Minority Accuracy: {test_minority_acc[0]}, Test Majority Accuracy: {test_majority_acc[0]}")

    for rd in range(1, NUM_ROUNDS+1):
        # Query
        query_idxs = strategy.query(args.nQuery)
        labelled_mask[query_idxs] = True
        strategy.update(labelled_mask)

        # Draw Conclusions: Gain additional information
        # Try adding this just to observe how well training the mask affects model
        strategy.train_prune([X_tr[i] for i in query_idxs], Y_tr[query_idxs], P_tr[query_idxs], X_val, Y_val, P_val, state_dict)
        print(f"Minority, Majority, Average Test Accuracy: {strategy.evaluate_model(loader_test)}")
        # Next Round: Train (with inductive bias) and Test
        state_dict = strategy.train(X_val, Y_val, P_val, verbose=False)
        test_minority_acc[rd], test_majority_acc[rd], test_average_acc[rd] = strategy.evaluate_model(loader_test)

        # Print and Clean up
        print(f"Round {rd}, Fraction of minority: {np.sum(Y_tr[query_idxs] != P_tr[query_idxs])}/{args.nQuery}, "
              f"Train Size: {np.sum(labelled_mask)}, Test Average Accuracy: {test_average_acc[rd]}, "
              f"Test Minority Accuracy: {test_minority_acc[rd]}, Test Majority Accuracy: {test_majority_acc[rd]}")
        torch.cuda.empty_cache()
        gc.collect()
