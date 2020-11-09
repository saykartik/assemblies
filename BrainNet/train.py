import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from network import BrainNet


np.set_printoptions(precision=4)


def plot_output_stats(all_true_y, all_pred_y_train, all_pred_y_test):
    plt.figure()
    plt.hist(all_true_y, bins=np.max(all_true_y)+1)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Frequency of ground truth labels')
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


def train_given_rule(X, y, meta_model, verbose=False, X_test=None, y_test=None):
    '''
    Trains a network using a fixed set of plasticity rules.
    '''
    all_rules = []
    test_accuracies = []
    train_accuracies = []
    all_true_y = []
    all_pred_y_train = []
    all_pred_y_test = []

    batch = 1
    for k in range(len(X)):
        inputs_numpy = X[k*batch:(k+1)*batch, :]
        labels_numpy = y[k*batch:(k+1)*batch]
        inputs = torch.from_numpy(inputs_numpy).double()
        labels = torch.from_numpy(labels_numpy).long()

        if k == 0:
            continue_ = False
        else:
            continue_ = True
        loss = meta_model(inputs, labels, 1, batch, continue_=continue_)

        if k == len(X) - 1 or (verbose and k % 500 == 0):
            print("Train on", k, " examples.")
            acc, pred_y_train = evaluate(X, y, meta_model.m, meta_model.forward_pass)
            train_accuracies.append(acc)
            print("Train Accuracy: {0:.4f}".format(acc))

            test_acc, pred_y_test = evaluate(X_test, y_test, meta_model.m, meta_model.forward_pass)
            test_accuracies.append(test_acc)
            print("Test Accuracy: {0:.4f}".format(test_acc))
                
            # Store all outputs and labels.
            for i in range(batch):
                all_true_y.append(labels_numpy[i])
                all_pred_y_train.append(pred_y_train[i])
                all_pred_y_test.append(pred_y_test[i])

    # Some data to plot and return.
    all_true_y = np.array(all_true_y, dtype=np.int32)
    all_pred_y_train = np.array(all_pred_y_train, dtype=np.int32)
    all_pred_y_test = np.array(all_pred_y_test, dtype=np.int32)
    other_stats = (all_true_y, all_pred_y_train, all_pred_y_test)
    plot_output_stats(all_true_y, all_pred_y_train, all_pred_y_test)

    return train_accuracies, test_accuracies, other_stats


def train_local_rule(X, y, meta_model, rule_epochs, epochs, batch, lr=1e-2, X_test=None, y_test=None, verbose=False):
    '''
    Meta-learns a set of plasticity rules on the given dataset.
    If fixed_rule == True, we only run GD on input layer and keep rule fixed
    Otherwise, apply GD to both input layer and local learning rule.
    '''
    meta_model.double()

    optimizer = optim.Adam(meta_model.parameters(), lr=lr, weight_decay=0.01)

    sz = len(X)

    # Stats to keep track of.
    running_loss = []
    all_train_acc = []
    all_test_acc = []
    all_true_y = []
    all_pred_y_train = []
    all_pred_y_test = []

    print("Starting Train")
    for epoch in range(1, rule_epochs + 1):
        X, y = shuffle(X, y)

        print('Outer epoch ', epoch)

        cur_losses = []
        train_accuracies = []
        test_accuracies = []
        for k in range(sz // batch):

            optimizer.zero_grad()

            inputs_numpy = X[k*batch:(k+1)*batch, :]
            labels_numpy = y[k*batch:(k+1)*batch]
            inputs = torch.from_numpy(inputs_numpy).double()
            labels = torch.from_numpy(labels_numpy).long()

            loss = meta_model(inputs, labels, epochs, batch)
            cur_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        if verbose or epoch == rule_epochs:
            acc, pred_y_train = evaluate(X, y, meta_model.m, meta_model.forward_pass)
            train_accuracies.append(acc)
            print("Train Accuracy: {0:.4f}".format(acc))
            
            if not (X_test is None):
                test_acc, pred_y_test = evaluate(X_test, y_test, meta_model.m, meta_model.forward_pass)
                test_accuracies.append(test_acc)
                print("Test Accuracy: {0:.4f}".format(test_acc))
                
            # Store all outputs and labels.
            for i in range(batch):
                all_true_y.append(labels_numpy[i])
                all_pred_y_train.append(pred_y_train[i])
                if not (X_test is None):
                    all_pred_y_test.append(pred_y_test[i])

        loss = np.mean(cur_losses)
        running_loss.append(loss.item())
        # print("LOSS:", np.mean(running_loss))
        print("Current loss:", loss.item())
        print("Mean loss so far:", np.mean(running_loss))

        all_train_acc.append(train_accuracies)
        all_test_acc.append(test_accuracies)

    # Some data to plot and return.
    all_true_y = np.array(all_true_y, dtype=np.int32)
    all_pred_y_train = np.array(all_pred_y_train, dtype=np.int32)
    all_pred_y_test = np.array(all_pred_y_test, dtype=np.int32)
    other_stats = (all_true_y, all_pred_y_train, all_pred_y_test)
    plot_output_stats(all_true_y, all_pred_y_train, all_pred_y_test)

    return running_loss, all_train_acc, all_test_acc, other_stats


def train_vanilla(X, y, model, epochs, batch, lr=1e-2):
    '''
    Trains a network using gradient descent (no plasticity rules involved).
    '''
    X, y = shuffle(X, y)

    model.double()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sz = len(X)
    criterion = nn.CrossEntropyLoss()

    print("INITIAL ACCURACY")
    acc, pred_y = evaluate(X, y, model.m, model)
    print("epoch 0", "Accuracy: {0:.4f}".format(acc))

    total_samples = 0
    samples = [total_samples]
    accuracies = [acc]

    running_loss = []
    for epoch in range(1, epochs + 1):

        cur_losses = []
        for k in range(sz//batch):

            inputs = X[k*batch:(k+1)*batch, :]
            labels = y[k*batch:(k+1)*batch]

            inputs = torch.from_numpy(inputs).double()
            labels = torch.from_numpy(labels).long()

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cur_losses.append(loss.item())

            total_samples += batch
            if total_samples % 1000 == 0:
                samples.append(total_samples)
                acc, pred_y = evaluate(X, y, model.m, model)
                accuracies.append(acc)

        running_loss.append(np.mean(cur_losses))
        if epoch % 1 == 0:
            print("Evaluating")
            acc, pred_y = evaluate(X, y, model.m, model)
            print("epoch ", epoch, "Loss: {0:.4f}".format(
                running_loss[-1]), "Accuracy: {0:.4f}".format(acc))

    print('Finished Training')
    return running_loss, samples, accuracies


def evaluate(X, y, num_labels, model_forward):
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

        for i in range(num_labels):
            print("Acc of class", i, ":{0:.4f}".format(ac[i] / (total[i] + 1e-6)))
    
    return acc, b
