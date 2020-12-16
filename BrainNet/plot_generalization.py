'''
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


# Process arguments.
parser = argparse.ArgumentParser()


results_dir = 'generalization_results/'

table_models = ['table_prepost', 'table_prepostcount', 'table_prepostpercent']
ann_models = ['reg_oneprepost', 'reg_oneprepostall', 'reg_allpostall']

property_friendlies = {
    'model': 'model',
    'nhl': 'num_hidden_layers',
    'hw': 'hidden_width',
    'dup': 'dataset_upstream',
    'uni': 'universal',
    'dsbp': 'downstream_backprop'
}

property_ranges = {
    'model': [*table_models, *ann_models],
    'nhl': [1, 2, 3],
    'hw': [100, 500],
    'dup': ['halfspace', 'relu'],
    'uni': [1, 0],
    'dsbp': [1, 0]
}


def _load_results_from_files(must_contain):
    all_files = os.listdir(results_dir)
    all_files = [fn for fn in all_files if must_contain in fn]
    results = []
    for fn in all_files:
        fp = os.path.join(results_dir, fn)
        with open(fp, 'rb') as f:
            cur_res = pickle.load(f)
        results.append((fn, cur_res))
    return results


def _print_mean_metrics(args):
    for parameter in property_ranges:
        for value in property_ranges[parameter]:
            if parameter == 'model':
                must_contain = value
            else:
                must_contain = '_' + parameter + str(value)
            
            results = _load_results_from_files(must_contain)
            if len(results) != 0:
                print(parameter, '=', value)
                print('Count:', len(results))
                final_accs_up = []
                final_accs_down = []
                
                for (exp_tag, stats) in results:
                    if stats[0] is not None:
                        stats_up = convert_multi_stats_uncertainty(stats[0])
                        final_accs_up.append(stats_up[4][-1])
                    stats_down = convert_multi_stats_uncertainty(stats[1])
                    final_accs_down.append(stats_down[4][-1])

                print(f'Mean final upstream accuracy: {np.mean(final_accs_up):.3f}')
                print(f'Mean final downstream accuracy: {np.mean(final_accs_down):.3f}')
                print()


def main(args):
    _print_mean_metrics(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
