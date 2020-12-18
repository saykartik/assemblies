'''
Tools for running various experiments.
Created by Basile Van Hoorick, November 2020.
'''

# Library imports.
import copy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pathlib
import torch
import torchvision
import torchvision.datasets
from sklearn.utils import shuffle

# Repository imports.
from BrainNet import BrainNet
from DataGenerator import random_halfspace_data, layer_relu_data
from FFBrainNet import FFBrainNet
from FFLocalNet import FFLocalNet
from FFLocalPlasticityRules.TableRule_PrePost import TableRule_PrePost
from FFLocalPlasticityRules.TableRule_PrePostCount import TableRule_PrePostCount
from FFLocalPlasticityRules.TableRule_PrePostPercent import TableRule_PrePostPercent
from FFLocalPlasticityRules.TableRule_PostCount import TableRule_PostCount
from FFLocalPlasticityRules.OneBetaANNRule_PrePost import OneBetaANNRule_PrePost
from FFLocalPlasticityRules.OneBetaANNRule_PrePostAll import OneBetaANNRule_PrePostAll
from FFLocalPlasticityRules.OneBetaANNRule_PostAll import OneBetaANNRule_PostAll
from FFLocalPlasticityRules.AllBetasANNRule_PostAll import AllBetasANNRule_PostAll
from LocalNetBase import Options, UpdateScheme
from network import LocalNet
from train import metalearn_rules, train_downstream


def quick_get_data(which, dim, N=10000, split=0.75, relu_k=1000):
    '''
    Quick, get some data!
    '''
    which = which.lower()

    if which == 'halfspace':
        X, y = random_halfspace_data(dim=dim, n=N)

    elif which == 'relu':
        X, y = layer_relu_data(dim, N, relu_k)
        class0_cnt = np.sum(y == 0)
        class1_cnt = np.sum(y == 1)
        total = class0_cnt + class1_cnt
        print(
            f'Class 0: {class0_cnt} ({class0_cnt * 100 / total:.1f}%)   Class 1: {class1_cnt} ({class1_cnt * 100 / total:.1f}%)')

    elif which == 'mnist':
        # NOTE: Argument N is ignored here.
        mnist_train = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=None)
        mnist_test = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=None)

        print('MNIST train:', len(mnist_train))
        print('MNIST test:', len(mnist_test))
        X_train = np.array([np.array(pair[0]) for pair in mnist_train]) / 255.0
        y_train = np.array([pair[1] for pair in mnist_train])
        X_test = np.array([np.array(pair[0]) for pair in mnist_test]) / 255.0
        y_test = np.array([pair[1] for pair in mnist_test])
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # print('Shuffling and reducing to train / test: 10000 / remainder, ignoring N...')
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)
        N_train = 10000
        N_test = int(N_train * (1 - split) / split)
        X_train = X_train[:N_train]
        y_train = y_train[:N_train]
        X_test = X_test[:N_test]
        y_test = y_test[:N_test]

    else:
        raise ValueError('Unknown or unused dataset: ' + which)

    if which != 'mnist':
        X_train = X[:int(N * split)]
        y_train = y[:int(N * split)]
        X_test = X[int(N * split):]
        y_test = y[int(N * split):]

    print('X_train:', X_train.shape)
    print('X_test:', X_test.shape)

    return X_train, y_train, X_test, y_test


def evaluate_brain(brain_fact, n,
                   dataset_up='halfspace', dataset_down='halfspace', downstream_backprop=False,
                   num_runs=1, num_rule_epochs=10, num_epochs_upstream=1, num_epochs_downstream=1,
                   min_upstream_acc=0.7, batch_size=100, learn_rate=1e-2,
                   data_size=10000, relu_k=1000, use_gpu=False):
    '''
    Evaluate a SINGLE network instance by meta-learning and then
    training on a reinitialized dataset of the same dimensionality.

    Args:
        brain_fact: Calling brain_fact() will create a new instance of the network under test.
        dataset_up: Upstream dataset class (halfspace / relu / mnist).
        dataset_down: If None, keep the same dataset instance. Otherwise, downstream dataset class.
        downstream_backprop: Use backprop for the direct GD layers downstream?
            Recommended False if dataset_down is None, since we assume layers
            with direct gradient descent to be trained upstream already.
        min_upstream_acc: Keep meta-learning until we find a good random initialization with
            this final test balanced accuracy.

    Returns:
        (multi_stats_up, multi_stats_down).
        Both are lists of length num_runs.
    '''
    multi_stats_up = []
    multi_stats_down = []

    for run in range(num_runs):
        print()
        print(f'Run {run + 1} / {num_runs}...')

        # Upstream.
        success = False
        while not success:
            brain = brain_fact()  # NOTE: Some initializations are unlucky.

            X_train, y_train, X_test, y_test = quick_get_data(
                dataset_up, n, N=data_size, relu_k=relu_k)
            print('Meta-learning on ' + dataset_up + '...')
            stats_up = metalearn_rules(
                X_train, y_train, brain, num_rule_epochs=num_rule_epochs,
                num_epochs=num_epochs_upstream, batch_size=batch_size, learn_rate=learn_rate,
                X_test=X_test, y_test=y_test, verbose=False, use_gpu=use_gpu)

            success = (stats_up[2][-1] >= min_upstream_acc)
            if not success:
                print(f'Final upstream test acc {stats_up[2][-1]:.4f} not high enough, retrying...')

        # Downstream.
        # NO rule transfer needed since we reuse the same network,
        # but just on a possibly altered dataset.
        if dataset_down is not None:
            X_train, y_train, X_test, y_test = quick_get_data(
                dataset_down, n, N=data_size, relu_k=relu_k)
            print('Training SAME brain instance on ' + dataset_down + '...')
        else:
            print('Training SAME brain instance on the same dataset instance...')
        stats_down = train_downstream(
            X_train, y_train, brain, num_epochs=num_epochs_downstream,
            batch_size=batch_size, vanilla=False, learn_rate=learn_rate,
            X_test=X_test, y_test=y_test, verbose=False,
            stats_interval=300, disable_backprop=not (downstream_backprop),
            use_gpu=use_gpu)

        # Save this run.
        multi_stats_up.append(stats_up)
        multi_stats_down.append(stats_down)

        print()

    return (multi_stats_up, multi_stats_down)


def evaluate_up_down(brain_up_fact, brain_down_fact, n_up, n_down,
                     dataset_up='halfspace', dataset_down='halfspace', downstream_backprop=False,
                     num_runs=1, num_rule_epochs=10, num_epochs_upstream=1, num_epochs_downstream=1,
                     num_downstream_subruns=1,
                     get_model=False, min_upstream_acc=0.7, batch_size=100, learn_rate=1e-2,
                     data_size=10000, relu_k=1000, use_gpu=False,
                     upstream_only=False, return_upstream_brains=False):
    '''
    Evaluates a PAIR of brains on the quality of meta-learning
    and rule interpretations by training with transferred rules.

    Args:
        brain_up_fact: Calling this will create a new instance of the network to meta-learn.
        brain_down_fact: Calling this will create a new instance of the network to train.
        dataset_up: Upstream dataset class (halfspace / relu / mnist).
        dataset_down: If None, keep the same dataset instance. Otherwise, downstream dataset class.
        downstream_backprop: Use backprop for the direct GD layers downstream?
            Recommended True, since the downstream weights will remain randomly initialized otherwise.
        min_upstream_acc: Keep meta-learning until we find a good random initialization with
            this final test balanced accuracy.

    Returns:
        (multi_stats_up, multi_stats_down) or ((multi_stats_up, multi_stats_down), brain_down).
        Both are lists of length num_runs.
    '''
    if (dataset_down is None) != (n_down is None):
        raise ValueError('The nullness of dataset_down does not agree with that of n_down.')

    multi_stats_up = []
    multi_stats_down = []  # Will remain empty if upstream_only.
    upstream_brains = []

    for run in range(num_runs):
        print()
        print(f'Run {run + 1} / {num_runs}...')

        # Upstream (once per run).
        success = False
        failures = 0
        while not success and failures < 3:
            brain_up = brain_up_fact()  # NOTE: Some initializations are unlucky.

            print('Meta-learning on ' + dataset_up + '...')
            X_train, y_train, X_test, y_test = quick_get_data(
                dataset_up, n_up, N=data_size, relu_k=relu_k)
            stats_up = metalearn_rules(
                X_train, y_train, brain_up, num_rule_epochs=num_rule_epochs,
                num_epochs=num_epochs_upstream, batch_size=batch_size, learn_rate=learn_rate,
                X_test=X_test, y_test=y_test, verbose=False, use_gpu=use_gpu)

            success = (stats_up[2][-1] >= min_upstream_acc)
            if not success:
                failures += 1
                print(
                    f'Final upstream test acc {stats_up[2][-1]:.4f} not high enough, retrying... (failures = {failures})')

        # Retain network e.g. for extracting rules later.
        if return_upstream_brains:
            upstream_brains.append(brain_up)
        
        if not upstream_only:

            # Downstream (subrun loop).
            for subrun in range(num_downstream_subruns):

                if num_downstream_subruns > 1:
                    print()
                    print(f'Run {run + 1} / {num_runs}...')
                    print(f'Subrun {subrun + 1} / {num_downstream_subruns}...')

                # Transfer rules.
                brain_down = brain_down_fact()
                if isinstance(brain_down, FFLocalNet):
                    # FF-ANN.
                    brain_down.copy_rules(brain_up, output_rule=brain_down.options.use_output_rule)
                else:
                    # RNN.
                    try:
                        if brain_down.options.use_graph_rule:
                            brain_down.set_rnn_rule(brain_up.get_rnn_rule())
                        if brain_down.options.use_output_rule:
                            brain_down.set_output_rule(brain_up.get_output_rule())
                    except:
                        print('FALLBACK: direct assignment of rules...')
                        if downstream_backprop:
                            print('=> WARNING: Rules might still be updated by GD this way')
                        brain_down.rnn_rule = brain_up.rnn_rule
                        brain_down.output_rule = brain_up.output_rule

                # Downstream (within subrun).
                if dataset_down is not None and n_down is not None:
                    print('Training NEW brain instance on ' + dataset_down + '...')
                    X_train, y_train, X_test, y_test = quick_get_data(
                        dataset_down, n_down, N=data_size, relu_k=relu_k)
                else:
                    print('Training NEW brain instance on the same dataset instance...')
                stats_down = train_downstream(
                    X_train, y_train, brain_down, num_epochs=num_epochs_downstream,
                    batch_size=batch_size, vanilla=False, learn_rate=learn_rate,
                    X_test=X_test, y_test=y_test, verbose=False,
                    stats_interval=300, disable_backprop=not(downstream_backprop),
                    use_gpu=use_gpu)

                # Save this subrun.
                multi_stats_down.append(stats_down)

        # Save this run.
        multi_stats_up.append(stats_up)

        print()

    if get_model:
        # Return latest downstream network.
        return (multi_stats_up, multi_stats_down), brain_down
    elif return_upstream_brains:
        # Return all upstream networks.
        return (multi_stats_up, multi_stats_down), upstream_brains
    else:
        return (multi_stats_up, multi_stats_down)


def evaluate_vanilla(brain_fact, dim, dataset='halfspace', num_runs=1, num_epochs=1,
                     batch_size=100, learn_rate=1e-2, data_size=10000, relu_k=1000,
                     use_gpu=False):
    '''
    Evaluates a brain using regular gradient descent and backprop.

    Returns:
        multi_stats: A list of length num_runs.
    '''
    multi_stats = []

    for run in range(num_runs):
        print()
        print(f'Run {run + 1} / {num_runs}...')

        brain = brain_fact()

        X_train, y_train, X_test, y_test = quick_get_data(dataset, dim, N=data_size, relu_k=relu_k)
        stats = train_downstream(
            X_train, y_train, brain, num_epochs=num_epochs,
            batch_size=batch_size, vanilla=True, learn_rate=learn_rate,
            X_test=X_test, y_test=y_test, verbose=False,
            stats_interval=300, disable_backprop=False, use_gpu=use_gpu)

        # Save this run.
        multi_stats.append(stats)

        print()

    return multi_stats


def evaluate_generalization(brain_up_fact, brain_down_fact, n_up, n_down, **kwargs):
    '''
    Legacy method.
    Evaluate the quality of meta-learning and rule interpretations by
    training a different network on a more complex dataset with transferred rules.
    '''
    kwargs['downstream_backprop'] = True
    return evaluate_up_down(brain_up_fact, brain_down_fact, n_up, n_down, **kwargs)


def convert_multi_stats_uncertainty(multi_stats):
    '''
    Merge and summarize stats from multiple runs into one tuple that
    tracks means and standard deviations over time.
    '''
    all_losses = np.array([s[0] for s in multi_stats])
    all_train_acc = np.array([s[1] for s in multi_stats])
    all_test_acc = np.array([s[2] for s in multi_stats])
    #     print('all_losses:', all_losses.shape)

    # Summarize by calculating things across the 'run' dimension.
    losses_mean = all_losses.mean(axis=0)
    train_acc_mean = all_train_acc.mean(axis=0)
    test_acc_mean = all_test_acc.mean(axis=0)
    losses_std = all_losses.std(axis=0)
    train_acc_std = all_train_acc.std(axis=0)
    test_acc_std = all_test_acc.std(axis=0)
    #     print('losses_mean:', losses_mean.shape)
    #     print('losses_std:', losses_std.shape)

    sample_counts = multi_stats[0][3]  # We assume that this is the same everywhere!
    other_stats = None  # I wouldn't bother to combine this.

    agg_stats = (losses_mean, losses_std, train_acc_mean, train_acc_std,
                 test_acc_mean, test_acc_std, sample_counts, other_stats)
    return agg_stats


def plot_curves(agg_stats_up, agg_stats_down, title_up, title_down, save_name='figs/default', no_downstream_loss=False):
    '''
    Plot upstream (optional) and downstream (required) learning curves of ONE model.
    If multiple runs were executed, the shaded areas indicate standard deviations.
    '''
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']  # NOTE: Only six colors if using seaborn.

    if len(agg_stats_down) == 5:
        # One run.
        if agg_stats_up is not None:
            (meta_losses, meta_train_acc, meta_test_acc, meta_sample_counts, meta_stats) = agg_stats_up
        (plas_losses, plas_train_acc, plas_test_acc, plas_sample_counts, plas_stats) = agg_stats_down
        plot_std = False

    else:
        # Multiple runs.
        if agg_stats_up is not None:
            (meta_losses, meta_losses_std, meta_train_acc, meta_train_acc_std,
             meta_test_acc, meta_test_acc_std, meta_sample_counts, meta_stats) = agg_stats_up
        (plas_losses, plas_losses_std, plas_train_acc, plas_train_acc_std,
         plas_test_acc, plas_test_acc_std, plas_sample_counts, plas_stats) = agg_stats_down
        plot_std = True

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot = upstream.
    if agg_stats_up is not None:
        ax[0].plot(meta_sample_counts, meta_losses, label='loss', color=default_colors[2])
        ax[0].plot(meta_sample_counts, meta_train_acc, label='train', color=default_colors[1])
        ax[0].plot(meta_sample_counts, meta_test_acc, label='test', color=default_colors[0])
        if plot_std:
            ax[0].fill_between(meta_sample_counts, meta_losses - meta_losses_std,
                               meta_losses + meta_losses_std, alpha=0.3, facecolor=default_colors[2])
            ax[0].fill_between(meta_sample_counts, meta_train_acc - meta_train_acc_std,
                               meta_train_acc + meta_train_acc_std, alpha=0.3, facecolor=default_colors[1])
            ax[0].fill_between(meta_sample_counts, meta_test_acc - meta_test_acc_std,
                               meta_test_acc + meta_test_acc_std, alpha=0.3, facecolor=default_colors[0])
        ax[0].set_xlabel('Cumulative number of training samples')
        ax[0].set_ylabel('Balanced accuracy / Loss')
        ax[0].set_xlim(0, meta_sample_counts[-1])
        ax[0].set_title(title_up)
        ax[0].legend()
    else:
        ax[0].set_visible(False)

    # Right plot = downstream.
    if not no_downstream_loss:
        ax[1].plot(plas_sample_counts[1:], plas_losses[1:], label='loss', color=default_colors[2])
    ax[1].plot(plas_sample_counts, plas_train_acc, label='train', color=default_colors[1])
    ax[1].plot(plas_sample_counts, plas_test_acc, label='test', color=default_colors[0])
    if plot_std:
        if not no_downstream_loss:
            ax[1].fill_between(plas_sample_counts[1:], plas_losses[1:] - plas_losses_std[1:],
                               plas_losses[1:] + plas_losses_std[1:], alpha=0.3, facecolor=default_colors[2])
        ax[1].fill_between(plas_sample_counts, plas_train_acc - plas_train_acc_std,
                           plas_train_acc + plas_train_acc_std, alpha=0.3, facecolor=default_colors[1])
        ax[1].fill_between(plas_sample_counts, plas_test_acc - plas_test_acc_std,
                           plas_test_acc + plas_test_acc_std, alpha=0.3, facecolor=default_colors[0])
    ax[1].set_xlabel('Cumulative number of training samples')
    ax[1].set_ylabel('Balanced accuracy / Loss')
    ax[1].set_xlim(0, plas_sample_counts[-1])
    ax[1].set_title(title_down)
    ax[1].legend()

    fig.tight_layout()

    # Create parent directory if needed.
    if not os.path.exists(pathlib.Path(save_name).parent):
        os.makedirs(pathlib.Path(save_name).parent)

    # Store and display graph.
    print('Saving figure to:', save_name)
    plt.savefig(save_name + '.pdf', dpi=192)
    plt.savefig(save_name + '.png', dpi=192)
    plt.show()

    # Print essential stats.
    print('Mean essential stats across all runs:')
    if agg_stats_up is not None:
        print(f'Last upstream loss: {meta_losses[-1]:.4f}')
        print(f'Last upstream train balanced accuracy: {meta_train_acc[-1]:.4f}')
        print(f'Last upstream test balanced accuracy: {meta_test_acc[-1]:.4f}')
    print(f'Last downstream loss: {plas_losses[-1]:.4f}')
    print(f'Last downstream train balanced accuracy: {plas_train_acc[-1]:.4f}')
    print(f'Last downstream test balanced accuracy: {plas_test_acc[-1]:.4f}')
    print()


def get_colors_styles(labels):
    # NOTE: Please feel free to modify this method to improve your figures.
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']  # NOTE: Only six colors if using seaborn.

    colors = copy.deepcopy(default_colors)
    styles = ['solid'] * len(labels)

    # Handle exceptions.
    #     if len(labels) == 8:
    #         colors = [default_colors[0]] * len(labels)
    #         colors[0] = default_colors[0]  # RNN
    #         colors[1] = colors[2] = colors[3] = default_colors[1]  # PrePost
    #         colors[4] = colors[5] = colors[6] = default_colors[2]  # PrePostCount
    #         colors[7] = default_colors[3]  # Vanilla
    #         styles[2] = styles[5] = 'dashed'
    #         styles[3] = styles[6] = 'dotted'

    return colors, styles


def plot_compare_models(all_stats_up, all_stats_down, labels, title_up, title_down, save_name='figs/default'):
    '''
    Plot upstream (optional) and downstream (required) curves of
    only one metric (test balanced accuracy) across MANY models.
    '''
    num_models = len(all_stats_up)
    assert (num_models == len(all_stats_down) and num_models == len(labels))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors, styles = get_colors_styles(labels)

    if len(labels) > len(colors):
        raise ValueError("Too many plots at once (we don't have that many colors)")

    has_metalearning = False
    for i in range(num_models):
        agg_stats_up = all_stats_up[i]
        agg_stats_down = all_stats_down[i]

        if len(agg_stats_down) == 5:
            # One run.
            if agg_stats_up is not None:
                (meta_losses, meta_train_acc, meta_test_acc,
                 meta_sample_counts, meta_stats) = agg_stats_up
                has_metalearning = True
            (plas_losses, plas_train_acc, plas_test_acc, plas_sample_counts, plas_stats) = agg_stats_down
            plot_std = False

        else:
            # Multiple runs.
            if agg_stats_up is not None:
                (meta_losses, meta_losses_std, meta_train_acc, meta_train_acc_std,
                 meta_test_acc, meta_test_acc_std, meta_sample_counts, meta_stats) = agg_stats_up
            (plas_losses, plas_losses_std, plas_train_acc, plas_train_acc_std,
             plas_test_acc, plas_test_acc_std, plas_sample_counts, plas_stats) = agg_stats_down
            plot_std = True

        if agg_stats_up is not None:
            ax[0].plot(meta_sample_counts, meta_test_acc * 100, label=labels[i],
                       color=colors[i], linestyle=styles[i])
        ax[1].plot(plas_sample_counts, plas_test_acc * 100, label=labels[i],
                   color=colors[i], linestyle=styles[i])
        if plot_std:
            if agg_stats_up is not None:
                ax[0].fill_between(meta_sample_counts, (meta_test_acc - meta_test_acc_std) * 100,
                                   (meta_test_acc + meta_test_acc_std) * 100, alpha=0.3, facecolor=colors[i],
                                   linestyle=styles[i])
            ax[1].fill_between(plas_sample_counts, (plas_test_acc - plas_test_acc_std) * 100,
                               (plas_test_acc + plas_test_acc_std) * 100, alpha=0.3, facecolor=colors[i], linestyle=styles[i])

    ax[0].set_xlabel('Cumulative number of training samples')
    ax[0].set_ylabel('Test balanced accuracy [%]')
    ax[0].set_title(title_up)
    if has_metalearning:
        ax[0].set_xlim(meta_sample_counts[0], meta_sample_counts[-1])
    ax[0].legend()
    ax[1].set_xlabel('Cumulative number of training samples')
    ax[1].set_ylabel('Test balanced accuracy [%]')
    ax[1].set_title(title_down)
    ax[1].set_xlim(plas_sample_counts[0], plas_sample_counts[-1])
    ax[1].legend()

    fig.tight_layout()

    # Create parent directory if needed.
    if not os.path.exists(pathlib.Path(save_name).parent):
        os.makedirs(pathlib.Path(save_name).parent)

    # Store and display graph.
    print('Saving figure to:', save_name)
    plt.savefig(save_name + '.pdf', dpi=192)
    plt.savefig(save_name + '.png', dpi=192)
    # plt.show()


def results_filepath(filename):
    """Return a file path for the supplied file name"""
    _RESULTS_DIR = 'results/'
    if not os.path.isdir(_RESULTS_DIR):
        os.makedirs(_RESULTS_DIR)
    filepath = os.path.join(_RESULTS_DIR, filename)
    return filepath


def rules_filepath(filename_prefix):
    """Return a file path for the supplied file name"""
    _RULES_DIR = 'rules/'
    if not os.path.isdir(_RULES_DIR):
        os.makedirs(_RULES_DIR)
    filepath = os.path.join(_RULES_DIR, filename)
    return filepath


def plots_filepath(filename):
    """Return a file path for the supplied file name"""
    _PLOTS_DIR = 'plots/'
    if not os.path.isdir(_PLOTS_DIR):
        os.makedirs(_PLOTS_DIR)
    filepath = os.path.join(_PLOTS_DIR, filename)
    return filepath
