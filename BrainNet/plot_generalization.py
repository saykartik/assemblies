'''
Summarize and visualize MNIST results.
Created by Basile Van Hoorick, December 2020.
'''

# Library imports.
import argparse
import datetime
import os
import pickle
import shutil
import sys

# Repository imports.
from eval_util import *


_DEFAULT_RESULTS_DIR = 'generalization_results/'
_DEFAULT_PLOTS_DIR = 'generalization_plots/'

table_models = ['table_prepost', 'table_prepostcount', 'table_prepostpercent']
ann_models = ['reg_oneprepost', 'reg_oneprepostall', 'reg_allpostall']
all_models = ['ff', 'rnn', *table_models, *ann_models]

property_friendlies = {
    'model': 'model',
    'nhl': 'num_hidden_layers',
    'hw': 'hidden_width',
    'dup': 'dataset_upstream',
    'uni': 'universal',
    'dsbp': 'downstream_backprop',
    'van': 'vanilla'
}

property_ranges = {
    'model': all_models,
    'nhl': [1, 2, 3],
    'hw': [100, 500],
    'dup': ['halfspace', 'relu'],
    'uni': [1, 0],
    'dsbp': [1, 0],
    'van': [1, 0]
}

model_legends = {
    'ff': 'GD',
    'rnn': 'RNN',
    'table_prepost': 'PrePost',
    'table_prepostcount': 'PrePostCount',
    'table_prepostpercent': 'PrePostPercent',
    'reg_oneprepost': 'ANNPrePost',
    'reg_oneprepostall': 'ANNPrePostAll',
    'reg_allpostall': 'ANNAllPostAll'
}


# Process arguments.
parser = argparse.ArgumentParser()

# General network architecture.
parser.add_argument('--results_dir', default=_DEFAULT_RESULTS_DIR, type=str,
                    help='Path to input directory containing pickle files with generated stats.')
parser.add_argument('--plots_dir', default=_DEFAULT_PLOTS_DIR, type=str,
                    help='Path to output directory for storing figures.')


def _get_property_from_tag(exp_tag, name):
    if name == 'model':
        # Exists in beginning of tag.
        for model_name in all_models:
            if exp_tag.startswith(model_name + '_'):
                return model_name

        raise ValueError('Model not found in ' + exp_tag)

    else:
        # Parse assuming '..._nameval_...' or '..._nameval'
        if '_' + name in exp_tag:
            start = exp_tag.index('_' + name) + len('_' + name)
        else:
            # Probably absent (= false) boolean flag.
            # raise ValueError('Parameter ' + name + ' not found in ' + exp_tag)
            return '0'

        if '_' in exp_tag[start:]:
            end = exp_tag[start:].index('_') + start
        else:  # End of string.
            end = len(exp_tag)

        if start == end:
            # Probably present (= true) boolean flag.
            result = '1'
        else:
            result = exp_tag[start:end]

        return result


def _readable_exp_tag(exp_tag):
    vanilla = (_get_property_from_tag(exp_tag, 'van') is '1')
    if vanilla:
        # Exclude irrelevant parameters but include vanilla.
        to_mention = ['model', 'nhl', 'hw', 'van']
    else:
        to_mention = list(property_friendlies.keys())
        to_mention.remove('van')

    result = '('
    for i, parameter in enumerate(to_mention):
        value = _get_property_from_tag(exp_tag, parameter)
        result += property_friendlies[parameter] + ' = ' + value
        if i < len(to_mention) - 1:
            result += ', '
    result += ')'

    return result


def _load_results_from_files(args, must_contain='', must_not_contain='', summarize=False):
    '''
    Args:
        must_contain: Experiment tag (i.e. file name) must contain all non-empty strings of this list.
        must_not_contain: Exclude experiment tags that contain at least one non-empty string of this list.
        summarize: If True, aggregate multiple runs into one mean and stddev.
    Returns:
        [(tag1, stats1), (tag2, stats2), ...].
    '''

    if not(isinstance(must_contain, list)):
        must_contain = [must_contain]
    if not(isinstance(must_not_contain, list)):
        must_not_contain = [must_not_contain]

    all_files = os.listdir(args.results_dir)
    all_files = [fn for fn in all_files if fn.endswith('.p')]
    for mc in must_contain:
        if len(mc) != 0:
            all_files = [fn for fn in all_files if mc in fn]
    for mnc in must_not_contain:
        if len(mnc) != 0:
            all_files = [fn for fn in all_files if mnc not in fn]
    results = []

    for fn in all_files:
        exp_tag = fn[:-2]
        fp = os.path.join(args.results_dir, fn)
        with open(fp, 'rb') as f:
            cur_res = pickle.load(f)

        if summarize:
            # Typically, stats_up = length 5 list, stats_down = length 10 list
            (stats_up, stats_down) = cur_res
            if stats_up is not None:
                stats_up = convert_multi_stats_uncertainty(stats_up)
            stats_down = convert_multi_stats_uncertainty(stats_down)
            cur_res = (stats_up, stats_down)

        results.append((exp_tag, cur_res))

    return results


def _print_mean_metrics(args):
    '''
    Prints the (marginal) mean essential metrics,
    when conditioning on just one parameter at a time.
    '''
    print('==== Total Mean Metrics Across Single Parameter Values ====')

    for parameter in property_ranges:
        for value in property_ranges[parameter]:
            must_contain = ''
            must_not_contain = ''
            if parameter == 'model':
                must_contain = value + '_'
            elif parameter in ['uni', 'dsbp', 'van']:
                # Boolean => filter differently.
                if value:
                    must_contain = '_' + parameter
                else:
                    must_not_contain = '_' + parameter
            else:
                must_contain = '_' + parameter + str(value)

            results = _load_results_from_files(
                args, must_contain=must_contain,
                must_not_contain=must_not_contain, summarize=True)
            if len(results) != 0:
                print(property_friendlies[parameter], '=', value)
                print('Count:', len(results))
                final_accs_up = []
                final_accs_down = []

                for (exp_tag, stats) in results:
                    if stats[0] is not None:
                        final_accs_up.append(stats[0][4][-1])
                    final_accs_down.append(stats[1][4][-1])

                if len(final_accs_up) != 0:
                    print(f'Mean final upstream accuracy: {np.mean(final_accs_up)*100:.1f}%')
                print(f'Mean final downstream accuracy: {np.mean(final_accs_down)*100:.1f}%')
                print()

    print()


def _print_top_configs(args):
    print('==== Top Parameter Configurations ====')

    results = _load_results_from_files(args, summarize=True)
    print('All count:', len(results))

    # Sort by final downstream test accuracy, descending.
    print('Sorting by final downstream test accuracy...')
    results.sort(key=lambda kv: -kv[1][1][4][-1])
    top_k = 16
    results = results[:top_k]

    for (exp_tag, stats) in results:
        print(_readable_exp_tag(exp_tag))
        if stats[0] is not None:
            print(f'Final upstream accuracy: {stats[0][4][-1]*100:.1f}%')
        print(f'Final downstream accuracy: {stats[1][4][-1]*100:.1f}%')
        print()

    print()


def _plot_sweeps(args):
    print('==== Generate Parameter Comparison Graphs ====')

    # Plot different models.
    for nhl in ['_nhl1', '_nhl2', '_nhl3']:
        for hw in ['_hw100', '_hw500']:
            must_contain = [nhl, hw, '_duphalfspace', '_uni']
            must_not_contain = ['_dsbp', 'table_prepostcount', 'reg_oneprepostall', 'reg_allpostall']
            results = _load_results_from_files(
                args, must_contain=must_contain, must_not_contain=must_not_contain, summarize=True)
            must_contain = [nhl, hw, '_van']
            results.append(*_load_results_from_files(
                args, must_contain=must_contain, summarize=True))
            reorder = [4, 1, 3, 2, 0]
            if len(results) >= len(reorder):
                results = [results[i] for i in reorder]
            stats_up = []
            stats_down = []
            labels = []
            for (exp_tag, stats) in results:
                model_name = _get_property_from_tag(exp_tag, 'model')
                # legend_text = model_legends[model_name]
                legend_text = model_name
                stats_up.append(stats[0])
                stats_down.append(stats[1])
                labels.append(legend_text)
            plot_compare_models(stats_up, stats_down, labels,
                                'Upstream meta-learning on halfspace',
                                'Downstream training on MNIST',
                                os.path.join(args.plots_dir, 'models' + nhl + hw))

    # Plot different hidden layer counts.

    # Plot different hidden layer sizes.

    print()


def main(args):

    # Set plotting style
    use_seaborn = True
    if use_seaborn:
        plt.style.use('seaborn')
        # We need to add some colors to seaborn's default cycler (which only hax 6)
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        default_colors += ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=default_colors)

        # Add a semi-transparent backgroud to the legend for legibility
        plt.rcParams['legend.frameon'] = 'True'
        
    _print_mean_metrics(args)
    _print_top_configs(args)
    _plot_sweeps(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
