import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from network import BrainNet
from tqdm import tqdm


np.set_printoptions(precision=4)


def plot_output_stats(all_true_y_train, all_pred_y_train, all_true_y_test, all_pred_y_test):
    plt.figure()
    plt.hist(all_true_y_train, bins=np.max(all_true_y_train)+1)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Frequency of ground truth labels on training data')
    plt.show()

    plt.figure()
    plt.hist(all_pred_y_train, bins=np.max(all_pred_y_train)+1)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Frequency of model predictions on training data')
    plt.show()

    if len(all_pred_y_test):
        plt.figure()
        plt.hist(all_pred_y_test, bins=np.max(all_pred_y_test)+1)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Frequency of model predictions on testing data')
        plt.show()


def metalearn_rules(X, y, meta_model, num_rule_epochs, num_epochs, batch_size, learn_rate=1e-2,
                    X_test=None, y_test=None, verbose=False):
    '''
    Meta-learns a set of plasticity rules on the given dataset and network.
    We apply gradient descent on the rules (and some layers, if desired).

    Args:
        X: 2D numpy array with training data.
        y: 1D numpy array with training labels.
        meta_model: Instance of model to meta-train with an implementation of forward().
        num_rule_epochs: Number of outer epochs to meta-learn for.
        num_epochs: Number of inner epochs to measure the loss produced by rules for.
        batch_size: Batch size for meta-learning. Evaluation is done with all data points at once.
        verbose: If True, print loss & accuracy stats per outer epoch.
            If False, only summarize at the end.

    Returns:
        (all_losses, all_train_acc, all_test_acc, sample_counts, other_stats).
        all_losses: (num_rule_epochs) numpy array with mean upstream training loss per outer epoch.
        all_train_acc: (num_rule_epochs) numpy array with downstream training accuracy per outer epoch.
        all_test_acc: (num_rule_epochs) numpy array with downstream test accuracy per outer epoch.
        sample_counts: Cumulative number of training elements seen at every measurement.
        other_stats: (all_true_y_train, all_pred_y_train, all_true_y_test, all_pred_y_test)
            where each element is a (len(X)) numpy array representing the last outer epoch.
    '''
    meta_model.double()
    optimizer = optim.Adam(meta_model.parameters(), lr=learn_rate, weight_decay=0.01)

    data_count = len(X)
    num_batches = data_count // batch_size

    # Stats to keep track of.
    total_samples = 0  # Counter for easy plotting.
    all_losses = []
    all_train_acc = []
    all_test_acc = []
    sample_counts = []

    if verbose:
        print("Start meta-learning over outer (rule) epochs...")

    # For each outer (rule) epoch...
    for outer_epoch in tqdm(range(1, num_rule_epochs + 1)):
        # Re-shuffle the data.
        X, y = shuffle(X, y)
        if X_test is not None:
            X_test, y_test = shuffle(X_test, y_test)

        # print('Outer (rule) epoch ', outer_epoch)
        cur_losses = []

        # Loop over small batches of samples.
        for k in range(num_batches):
            optimizer.zero_grad()

            inputs_numpy = X[k*batch_size:(k+1)*batch_size]
            labels_numpy = y[k*batch_size:(k+1)*batch_size]
            inputs = torch.from_numpy(inputs_numpy).double()
            labels = torch.from_numpy(labels_numpy).long()
            total_samples += batch_size

            # Run inner epochs and update weights according to current rule.
            # NOTE: This resets the weights first.
            loss = meta_model(inputs, labels, num_epochs, batch_size)
            cur_losses.append(loss.item())

            # Update rules to reduce loss.
            loss.backward()
            optimizer.step()

        # Evaluate current performance over training data.
        train_acc, pred_y_train = evaluate(
            X, y, meta_model.m, meta_model.forward_pass, verbose=verbose)
        if verbose:
            print("Train accuracy: {0:.4f}".format(train_acc))

        # Evaluate current performance over test data.
        if X_test is not None:
            test_acc, pred_y_test = evaluate(
                X_test, y_test, meta_model.m, meta_model.forward_pass, verbose=verbose)
            if verbose:
                print("Test accuracy: {0:.4f}".format(test_acc))
        else:
            test_acc, pred_y_test = -1.0, None

        # Update loss.
        loss = np.mean(cur_losses)
        all_losses.append(loss)
        if verbose:
            print("Current loss: {0:.4f}".format(loss.item()))
            print("Mean loss so far: {0:.4f}".format(np.mean(all_losses)))

        all_train_acc.append(train_acc)
        all_test_acc.append(test_acc)
        sample_counts.append(total_samples)

    # Store all outputs and labels from the last outer epoch.
    all_losses = np.array(all_losses)
    all_train_acc = np.array(all_train_acc)
    all_true_y_train = np.array(y, dtype=np.int32)
    all_pred_y_train = np.array(pred_y_train, dtype=np.int32)
    if X_test is not None:
        all_test_acc = np.array(all_test_acc)
        all_true_y_test = np.array(y_test, dtype=np.int32)
        all_pred_y_test = np.array(pred_y_test, dtype=np.int32)
    else:
        all_test_acc = all_true_y_test = all_pred_y_test = np.empty(0)
    sample_counts = np.array(sample_counts, dtype=np.int32)

    # Plot and return.
    other_stats = (all_true_y_train, all_pred_y_train, all_true_y_test, all_pred_y_test)
    if verbose:
        plot_output_stats(*other_stats)

    print("Last loss: {0:.4f}".format(all_losses[-1]))
    print("Last train accuracy: {0:.4f}".format(all_train_acc[-1]))
    print("Last test accuracy: {0:.4f}".format(all_test_acc[-1]))

    return (all_losses, all_train_acc, all_test_acc, sample_counts, other_stats)


def train_downstream(X, y, model, num_epochs, batch_size, vanilla=False, learn_rate=1e-2,
                     X_test=None, y_test=None, verbose=False, stats_interval=500):
    '''
    If vanilla is False:
    Trains the network weights for one epoch using a fixed set of plasticity rules.
    Gradient descent is applied on some layers only, depending on the model's options.
    NOTE: The arguments batch_size and learn_rate are unused for rule-based learning.

    If vanilla is True:
    Trains a network using gradient descent (no plasticity rules involved) for several epochs.
    NOTE: Ensure that the model contains torch Parameters,
    otherwise the weights will not change!

    Args:
        X: 2D numpy array with training data.
        y: 1D numpy array with training labels.
        model: Either a custom rule-based network or a regular torch module instance.
        verbose: If True, print loss & accuracy stats every stats_interval data points.
            If False, only summarize at the end.
        stats_interval: If > 0, number of samples to process in-between subsequent
            accuracy measurements (also called sub-epochs).

    Returns:
        (all_losses, all_train_acc, all_test_acc, sample_counts, other_stats).
        all_losses: (num_epochs * num_batches / stats_interval) numpy array with mean training loss
            per sub-epoch evaluation.
        all_train_acc: (num_epochs * num_batches / stats_interval) numpy array with mean training accuracy
            per sub-epoch evaluation.
        all_test_acc: (num_epochs * num_batches / stats_interval) numpy array with mean test accuracy
            per sub-epoch evaluation.
        sample_counts: Cumulative number of training elements seen at every measurement.
        other_stats: (all_true_y_train, all_pred_y_train, all_true_y_test, all_pred_y_test)
            where each element is a (num_batches) numpy array representing every sample
            from the last sub-epoch evaluation.

    NOTE: The returned stats are always evaluated over the *whole* datasets,
    even though they are calculated multiple times per training epoch.
    '''
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    criterion = nn.CrossEntropyLoss()

    data_count = len(X)
    if vanilla:
        num_batches = data_count // batch_size
    else:
        batch_size = 1
        num_batches = data_count

    if vanilla:
        # Backprop-based.
        train_acc, _ = evaluate(X, y, model.m, model, verbose=verbose)
        print("INITIAL train accuracy: {0:.4f}".format(train_acc))
        if X_test is not None:
            test_acc, _ = evaluate(X_test, y_test, model.m, model, verbose=verbose)
            print("INITIAL test accuracy: {0:.4f}".format(test_acc))
        else:
            test_acc = -1.0
    
    else:
        # Rule-based; reset weights first by calling forward once.
        loss = model(torch.from_numpy(inputs[0:1]).double(),
                     torch.from_numpy(labels[0:1]).long(), 1, 1, continue_=False)
        train_acc, _ = evaluate(X, y, model.m, model.forward_pass, verbose=verbose)
        print("INITIAL train accuracy: {0:.4f}".format(train_acc))
        if X_test is not None:
            test_acc, _ = evaluate(X_test, y_test, model.m, model.forward_pass, verbose=verbose)
            print("INITIAL test accuracy: {0:.4f}".format(test_acc))
        else:
            test_acc = -1.0

    # Stats to keep track of.
    total_samples = 0  # Counter for easy plotting.
    all_losses = [0.0]
    all_train_acc = [train_acc]
    all_test_acc = [test_acc]
    sample_counts = [total_samples]

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch} / {num_epochs} ...')

        # Re-shuffle the data.
        X, y = shuffle(X, y)
        if X_test is not None:
            X_test, y_test = shuffle(X_test, y_test)
            
        cur_losses = []

        # Loop over all batches within this epoch.
        for k in tqdm(range(num_batches)):
            inputs_numpy = X[k*batch_size:(k+1)*batch_size]
            labels_numpy = y[k*batch_size:(k+1)*batch_size]
            inputs = torch.from_numpy(inputs_numpy).double()
            labels = torch.from_numpy(labels_numpy).long()
            total_samples += batch_size

            if vanilla:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            else:
                if k == 0:
                    continue_ = False  # Reset weights.
                else:
                    continue_ = True
                loss = model(inputs, labels, 1, batch_size, continue_=continue_)

            cur_losses.append(loss.item())

            # Periodically calculate stats.
            if k == num_batches - 1 or (stats_interval > 0 and
                                        total_samples % stats_interval < batch_size):
                
                # Evaluate current performance over training data.
                if vanilla:
                    train_acc, pred_y_train = evaluate(X, y, model.m, model, verbose=verbose)
                else:
                    train_acc, pred_y_train = evaluate(
                        X, y, model.m, model.forward_pass, verbose=verbose)
                all_train_acc.append(train_acc)
                if verbose:
                    print(f'Step {k+1} / {num_batches}')
                    print("Train accuracy: {0:.4f}".format(train_acc))

                # Evaluate current performance over test data.
                if X_test is not None:
                    if vanilla:
                        test_acc, pred_y_test = evaluate(
                            X_test, y_test, model.m, model, verbose=verbose)
                    else:
                        test_acc, pred_y_test = evaluate(
                            X_test, y_test, model.m, model.forward_pass, verbose=verbose)
                    all_test_acc.append(test_acc)
                    if verbose:
                        print("Test accuracy: {0:.4f}".format(test_acc))

                # Update loss.
                loss = np.mean(cur_losses)
                all_losses.append(loss)
                if verbose:
                    print("Current loss: {0:.4f}".format(loss.item()))
                    print("Mean loss so far: {0:.4f}".format(np.mean(all_losses)))

                sample_counts.append(total_samples)
                cur_losses = []

        print()

    # Store all outputs and labels from the last sub-epoch evaluation.
    all_losses = np.array(all_losses)
    all_train_acc = np.array(all_train_acc)
    all_true_y_train = np.array(y, dtype=np.int32)
    all_pred_y_train = np.array(pred_y_train, dtype=np.int32)
    if X_test is not None:
        all_test_acc = np.array(all_test_acc)
        all_true_y_test = np.array(y_test, dtype=np.int32)
        all_pred_y_test = np.array(pred_y_test, dtype=np.int32)
    else:
        all_test_acc = all_true_y_test = all_pred_y_test = np.empty(0)
    sample_counts = np.array(sample_counts, dtype=np.int32)

    # Plot and return.
    other_stats = (all_true_y_train, all_pred_y_train, all_true_y_test, all_pred_y_test)
    if verbose:
        plot_output_stats(*other_stats)

    print("Last loss: {0:.4f}".format(all_losses[-1]))
    print("Last train accuracy: {0:.4f}".format(all_train_acc[-1]))
    print("Last test accuracy: {0:.4f}".format(all_test_acc[-1]))

    return (all_losses, all_train_acc, all_test_acc, sample_counts, other_stats)


def evaluate(X, y, num_labels, model_forward, verbose=True):
    '''
    X, y: One batch of inputs and ground truths.
    Returns mean accuracy and array of predictions for this batch (i.e. dataset).
    '''
    ac = [0] * num_labels
    total = [0] * num_labels
    with torch.no_grad():

        correct = 0

        outputs = model_forward(torch.from_numpy(X).double())
        b = np.argmax(outputs, axis=1).numpy()

        for i in range(len(b)):
            total[y[i]] += 1
            if b[i] == y[i]:
                ac[y[i]] += 1

        correct = np.sum(y == b)

        acc = correct / sum(total)

        if verbose:
            for i in range(num_labels):
                print("Acc of class", i, ": {0:.4f}".format(ac[i] / (total[i] + 1e-6)))

    return acc, b
