'''
Display, summarize, and compare meta-learned plasticity rules.
Created by Basile Van Hoorick, December 2020.
'''

# Library imports.
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import sys

# Repository imports.
from eval_util import *

_DEFAULT_RESULTS_DIR = r'D:\Development\CVR Data\Plasticity Rules\inspect_results'
_DEFAULT_RULES_DIR = r'D:\Development\CVR Data\Plasticity Rules\rules'
_DEFAULT_PLOTS_DIR = 'inspect_plots/'

table_models = ['table_prepost', 'table_prepostcount', 'table_prepostpercent']
ann_models = ['reg_oneprepost', 'reg_oneprepostall', 'reg_allpostall']
all_models = ['ff', 'rnn', *table_models, *ann_models]

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
parser.add_argument('--rules_dir', default=_DEFAULT_RULES_DIR, type=str,
                    help='Path to input directory containing pickle files with plasticity rules.')
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


def _load_data_from_files(args, must_contain='', must_not_contain='', summarize=True):
    '''
    Args:
        must_contain: Experiment tag (i.e. file name) must contain all non-empty strings of this list.
        must_not_contain: Exclude experiment tags that contain at least one non-empty string of this list.
        summarize: If True, aggregate multiple runs into one mean and stddev.
    Returns:
        [(tag1, stats1, rules1), (tag2, stats2, rules2), ...].
    '''

    if not(isinstance(must_contain, list)):
        must_contain = [must_contain]
    if not(isinstance(must_not_contain, list)):
        must_not_contain = [must_not_contain]

    all_files = os.listdir(args.rules_dir)
    all_files = [fn for fn in all_files if fn.endswith('.p')]
    for mc in must_contain:
        if len(mc) != 0:
            all_files = [fn for fn in all_files if mc in fn]
    for mnc in must_not_contain:
        if len(mnc) != 0:
            all_files = [fn for fn in all_files if mnc not in fn]
    results = []

    for fn in all_files:
        exp_tag = fn[:-8]
        stats_fp = os.path.join(args.results_dir, exp_tag + '.p')
        rules_fp = os.path.join(args.rules_dir, exp_tag + '_rules.p')
        if not(os.path.isfile(stats_fp)) or not(os.path.isfile(rules_fp)):
            continue

        with open(stats_fp, 'rb') as f:
            (stats_up, _) = pickle.load(f)
        with open(rules_fp, 'rb') as f:
            all_rules = pickle.load(f)  # [(hidden1, output1), (hidden2, output2), ...]
        
        if summarize:
            # Typically, stats_up = length 5 list.
            stats_up = convert_multi_stats_uncertainty(stats_up)

        results.append((exp_tag, stats_up, all_rules))

    return results


def _print_rule(rule, desc):
    print(desc)
    print(np.array(rule))
    print()


def _print_rules(args):
    '''
    Simply prints rules for prepost.
    '''
    results = _load_data_from_files(args, must_contain='table_prepost_')
    
    # Loop over all experiments and all runs.
    for (exp_tag, stats, rules) in results:
        
        print(exp_tag)
        if '_gr' in exp_tag:
            print('=> Has hidden layer rule')
        if '_or' in exp_tag:
            print('=> Has output layer rule')
        
        # for rule_set in rules:
        #     (hidden_rule, output_rule) = rule_set
        #     if hidden_rule is not None:
        #         _print_rule(hidden_rule, 'Hidden:')
        #     if output_rule is not None:
        #         _print_rule(output_rule, 'Output:')

        if rules[0][0] is not None:
            all_hidden = [np.array(rs[0]) for rs in rules]
            hidden_mean = np.mean(all_hidden, axis=0)
            hidden_std = np.std(all_hidden, axis=0)
            _print_rule(hidden_mean, 'Mean hidden:')
            _print_rule(hidden_std, 'Stddev hidden:')
        if rules[0][1] is not None:
            all_output = [np.array(rs[1]) for rs in rules]
            output_mean = np.mean(all_output, axis=0)
            output_std = np.std(all_output, axis=0)
            _print_rule(output_mean, 'Mean output:')
            _print_rule(output_std, 'Stddev output:')

        print()


def _plot_rules(args):
    '''
    Plots rules for prepostcount or prepostpercent as they vary over layer firing count.
    '''
    must_contains = ['table_prepostcount_', 'table_prepostpercent_']
    for must_contain in must_contains:
        results = _load_data_from_files(args, must_contain=must_contain)
        
        # Loop over all experiments and all runs.
        for (exp_tag, stats, rules) in results:
            
            print(exp_tag)
            if '_gr' in exp_tag:
                print('=> Has hidden layer rule')
            if '_or' in exp_tag:
                print('=> Has output layer rule')
            
            # for rule_set in rules:
            #     (hidden_rule, output_rule) = rule_set
            #     if hidden_rule is not None:
            #         _print_rule(hidden_rule, 'Hidden:')
            #     if output_rule is not None:
            #         _print_rule(output_rule, 'Output:')

            fig = plt.figure(figsize=(6, 4))

            if rules[0][0] is not None:
                all_hidden = [np.array(rs[0]) for rs in rules]
                hidden_mean = np.mean(all_hidden, axis=0)
                hidden_std = np.std(all_hidden, axis=0)
                _print_rule(hidden_mean, 'Mean hidden:')
                _print_rule(hidden_std, 'Stddev hidden:')

                if 'percent' in must_contain:
                    xs = np.linspace(0, 100, len(hidden_mean[0, 0]))
                else:
                    xs = np.arange(len(hidden_mean[0, 0]))
                for i in range(2):
                    for j in range(2):
                        label = 'Hidden $(' + str(i) + ', ' + str(j) + ')$'
                        plt.plot(xs, hidden_mean[i, j], label=label)

            if rules[0][1] is not None:
                all_output = [np.array(rs[1]) for rs in rules]
                output_mean = np.mean(all_output, axis=0)
                output_std = np.std(all_output, axis=0)
                _print_rule(output_mean, 'Mean output:')
                _print_rule(output_std, 'Stddev output:')

                if 'percent' in must_contain:
                    xs = np.linspace(0, 100, len(output_mean[0, 0]))
                else:
                    xs = np.arange(len(output_mean[0, 0]))
                for i in range(2):
                    for j in range(2):
                        label = 'Output $(' + str(i) + ', ' + str(j) + ')$'
                        plt.plot(xs, output_mean[i, j], label=label)
            
            model_name = _get_property_from_tag(exp_tag, 'model')
            dataset = _get_property_from_tag(exp_tag, 'dup')
            if 'percent' in must_contain:
                plt.xlabel('Fraction of incoming firing nodes [%]')
            else:
                plt.xlabel('Number of incoming firing nodes')
            plt.ylabel(r'$\beta$')
            title = f'Plasticity rules for {model_legends[model_name]} on {dataset}'
            plt.title(title)
            plt.legend()
            fig.tight_layout()
            
            save_path = os.path.join(args.plots_dir, exp_tag)
            print('Saving figure to:', save_path)
            plt.savefig(save_path + '.pdf', dpi=192)
            plt.savefig(save_path + '.png', dpi=192)
            # plt.show()

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
        
    _print_rules(args)
    _plot_rules(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


