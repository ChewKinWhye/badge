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
    X_tr, Y_tr, P_tr, X_val, Y_val, P_val, X_te, Y_te, P_te, num_classes, num_attributes, handler, target_resolution = get_data(args.dataset, args.data_dir, args.spurious_strength,
                                               args.seed)
    if args.architecture == "ViT":
        target_resolution = (224, 224)
    if args.dataset in ["mcdominoes", "spuco"]:
        # For these two datasets, all minority groups are the same size. The evaluation should be done on all the minority groups
        test_group = "minority"
    else:
        # For the other datasets, some minority groups have smaller sizes than others. The evaluation should be done on the worst group
        test_group = "worst"

    # Initial labelled pool, Randomly select nStart indices to label
    labelled_mask = np.zeros(len(X_tr))
    labelled_mask[np.random.choice(len(X_tr), args.nStart, replace=False)] = 1

    # Acquisition Algorithm
    if args.alg == 'rand': # random sampling
        strategy = RandomSampling(X_tr, Y_tr, P_tr, labelled_mask.astype(bool), handler, num_classes, num_attributes, args.num_epochs, target_resolution, test_group, args)
    elif args.alg == 'conf': # confidence-based sampling
        strategy = LeastConfidence(X_tr, Y_tr, P_tr, labelled_mask.astype(bool), handler, num_classes, num_attributes, args.num_epochs, target_resolution, test_group, args)
    elif args.alg == 'badge': # batch active learning by diverse gradient embeddings
        strategy = BadgeSampling(X_tr, Y_tr, P_tr, labelled_mask.astype(bool), handler, num_classes, num_attributes, args.num_epochs, target_resolution, test_group, args)
    elif args.alg == 'coreset': # coreset sampling
        strategy = CoreSet(X_tr, Y_tr, P_tr, labelled_mask.astype(bool), handler, num_classes, num_attributes, args.num_epochs, target_resolution, test_group, args)
    else:
        print('Choose a valid acquisition function.')
        raise ValueError

    # Stats
    NUM_ROUNDS = (args.nEnd - args.nStart) // args.nQuery
    test_average_acc, test_minority_acc, test_majority_acc = np.zeros(NUM_ROUNDS+1), np.zeros(NUM_ROUNDS+1), np.zeros(NUM_ROUNDS+1)
    loader_test = DataLoader(handler(X_te, torch.Tensor(Y_te).long(), torch.Tensor(P_te).long(), isTrain=False, target_resolution=target_resolution),
                            shuffle=False, batch_size=args.batch_size)
    # Round 0 Train and Test
    strategy.train(X_val, Y_val, P_val, verbose=True) # Print out first round of training
    test_average_acc[0], test_minority_acc[0], test_majority_acc[0] = strategy.evaluate_model(loader_test)
    print(f"Round 0, Train Size: {np.sum(labelled_mask)}, Test Average Accuracy: {test_average_acc[0]}, "
          f"Test Minority/Worst Accuracy: {test_minority_acc[0]}, Test Majority/Best Accuracy: {test_majority_acc[0]}")
    state_dict = None
    for rd in range(1, NUM_ROUNDS+1):
        # Query
        query_idxs = strategy.query(args.nQuery)
        labelled_mask[query_idxs] = rd + 1
        strategy.update(labelled_mask)

        # Draw Conclusions: Gain additional information
        if args.method == "meta":
            state_dict = strategy.train_MAML(labelled_mask, X_val, Y_val, P_val, verbose=False)
            print(f"Test: Average Accuracy, Minority/Worst, Majority/Best: {strategy.evaluate_model(loader_test)}")

        # Next Round: Train and Test
        strategy.train(X_val, Y_val, P_val, state_dict, verbose=False)
        test_average_acc[rd], test_minority_acc[rd], test_majority_acc[rd] = strategy.evaluate_model(loader_test)

        # Print and Clean up
        print(f"Round {rd}, Fraction of minority: {np.sum(Y_tr[query_idxs] != P_tr[query_idxs])}/{args.nQuery}, "
              f"Train Size: {np.sum(labelled_mask.astype(bool))}, Test Average Accuracy: {test_average_acc[rd]}, "
              f"Test Minority/Worst Accuracy: {test_minority_acc[rd]}, Test Majority/Best Accuracy: {test_majority_acc[rd]}")
        torch.cuda.empty_cache()
        gc.collect()
