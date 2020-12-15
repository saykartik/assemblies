'''
An alternative to fiddling with Jupyter notebooks.
Run experiments with repeated trials at one specific hyperparameter configuration,
and store the output statistics to pickle files for later visualization.
In every case, we perform both upstream meta-learning and downstream training
on separate network instances with transferred rules.
Created by Basile Van Hoorick, December 2020.
'''

# Library imports.
import argparse
import datetime
import pickle
import shutil
import sys

# Repository imports.
from eval_util import *


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Process arguments.
parser = argparse.ArgumentParser()

# General network architecture.
parser.add_argument('--model', default='rnn', type=str,
                    help='Type of architecture and rule to use (rnn / '
                    'table_prepost / table_prepostcount / table_prepostpercent / table_postcount / '
                    'reg_oneprepost / reg_oneprepostall / reg_onepostall / reg_allpostall / ff) '
                    '(default: rnn).')
parser.add_argument('--use_graph_rule', default=True, type=str2bool,
                    help='Meta-learn and use plasticity rules for hidden layers instead of '
                    'backprop (default: True).')
parser.add_argument('--use_output_rule', default=True, type=str2bool,
                    help='Meta-learn and use plasticity rules for output layer instead of '
                    'backprop (default: True).')
parser.add_argument('--num_hidden_layers', default=2, type=int,
                    help='Number of hidden layers for FF, or T (= rounds + 1) for RNN '
                    '(default: 2).')
parser.add_argument('--hidden_width', default=100, type=int,
                    help='Width of every hidden layer (or number of vertices in graph for RNN) '
                    '(default: 100).')
parser.add_argument('--conn_prob', default=0.5, type=float,
                    help='Probability of synapse existence between every pair of eligible neurons '
                    '(default: 0.5).')
parser.add_argument('--proj_cap', default=50, type=int,
                    help='Upper bound number of firing neurons at every hidden layer '
                    '(or time step) to this value; recommended = width // 2 (default: 50).')
parser.add_argument('--universal', default=True, type=str2bool,
                    help='Use universal plasticity rule instead of per-later specialized ones. '
                    '(default: False).')

# Dataset and dimensionality.
parser.add_argument('--dataset_up', default='halfspace', type=str,
                    help='Meta-learning dataset name (halfspace / relu / mnist) '
                    '(default: halfspace).')
parser.add_argument('--dataset_down', default='halfspace', type=str,
                    help='Downstream task dataset name (halfspace / relu / mnist) '
                    '(default: halfspace).')
parser.add_argument('--n_up', default=10, type=int,
                    help='Upstream dataset size (a.k.a. dimensionality of input layer) '
                    '(default: 10).')
parser.add_argument('--n_down', default=20, type=int,
                    help='Downstream dataset size (a.k.a. dimensionality of input layer) '
                    '(default: 20).')
parser.add_argument('--m_up', default=2, type=int,
                    help='Upstream label count (a.k.a. dimensionality of output layer) '
                    '(default: 2).')
parser.add_argument('--m_down', default=2, type=int,
                    help='Downstream label count (a.k.a. dimensionality of output layer) '
                    '(default: 2).')
parser.add_argument('--data_size', default=4000, type=int,
                    help='Total number of elements in halfspace or relu dataset (default: 4000). '
                    'Note that the train/test split after generation is 0.75.')

# Training and loss.
parser.add_argument('--num_runs', default=5, type=int,
                    help='Number of times to repeat the experiment for more reliable statistics. '
                    '(default: 10).')
parser.add_argument('--num_rule_epochs', default=100, type=int,
                    help='Number of upstream outer epochs. '
                    '(default: 100).')
parser.add_argument('--num_epochs_upstream', default=1, type=int,
                    help='Number of upstream inner epochs. '
                    '(default: 1).')
parser.add_argument('--num_epochs_downstream', default=1, type=int,
                    help='Number of downstream epochs. '
                    '(default: 1).')
parser.add_argument('--batch_size', default=100, type=int,
                    help='Mini-batch size at all times. '
                    '(default: 100).')
parser.add_argument('--learn_rate', default=1e-2, type=float,
                    help='Learning rate at all times. '
                    '(default: 1e-2).')
parser.add_argument('--vanilla', default=False, type=str2bool,
                    help='Discard rule options and train everything with regular backprop '
                    '(default: False).')
parser.add_argument('--downstream_backprop', default=True, type=str2bool,
                    help='Train with gradient descent and backpropagation on the downstream '
                    'network instance (NOTE: not supported by RNN) (default: True).')

# Miscellaneous.
parser.add_argument('--ignore_if_exist', default=True, type=str2bool,
                    help='If True, do not run experiment if the results file already exists.')


def _create_table_rules(args):
    if args.model == 'table_prepost':
        if args.universal:
            hl_rules = TableRule_PrePost()
        else:
            hl_rules = [TableRule_PrePost() for _ in range(args.num_hidden_layers - 1)]
        output_rule = TableRule_PrePost()

    elif args.model == 'table_prepostcount':
        if args.universal:
            hl_rules = TableRule_PrePostCount()
        else:
            hl_rules = [TableRule_PrePostCount() for _ in range(args.num_hidden_layers - 1)]
        output_rule = TableRule_PrePostCount()

    elif args.model == 'table_prepostpercent':
        if args.universal:
            hl_rules = TableRule_PrePostPercent()
        else:
            hl_rules = [TableRule_PrePostPercent() for _ in range(args.num_hidden_layers - 1)]
        output_rule = TableRule_PrePostPercent()

    elif args.model == 'table_postcount':
        if args.universal:
            hl_rules = TableRule_PostCount()
        else:
            hl_rules = [TableRule_PostCount() for _ in range(args.num_hidden_layers - 1)]
        output_rule = TableRule_PostCount()

    else:
        raise ValueError('Unknown model / table rule type:', args.model)

    # Discard rules that are not needed.
    if not args.use_graph_rule:
        hl_rules = None
    if not args.use_output_rule:
        output_rule = None

    return hl_rules, output_rule


def _create_regression_rules(args):
    if args.model == 'reg_oneprepost':
        if args.universal:
            hl_rules = OneBetaANNRule_PrePost()
        else:
            hl_rules = [OneBetaANNRule_PrePost() for _ in range(args.num_hidden_layers - 1)]
        output_rule = OneBetaANNRule_PrePost()

    elif args.model == 'reg_oneprepostall':
        if args.universal:
            hl_rules = OneBetaANNRule_PrePostAll()
        else:
            hl_rules = [OneBetaANNRule_PrePostAll() for _ in range(args.num_hidden_layers - 1)]
        output_rule = OneBetaANNRule_PrePostAll()

    elif args.model == 'reg_onepostall':
        if args.universal:
            hl_rules = OneBetaANNRule_PostAll()
        else:
            hl_rules = [OneBetaANNRule_PostAll() for _ in range(args.num_hidden_layers - 1)]
        output_rule = OneBetaANNRule_PostAll()

    elif args.model == 'reg_allpostall':
        if args.universal:
            hl_rules = AllBetasANNRule_PostAll()
        else:
            hl_rules = [AllBetasANNRule_PostAll() for _ in range(args.num_hidden_layers - 1)]
        output_rule = AllBetasANNRule_PostAll()

    else:
        raise ValueError('Unknown model / regression rule type:', args.model)

    # Discard rules that are not needed.
    if not args.use_graph_rule:
        hl_rules = None
    if not args.use_output_rule:
        output_rule = None

    return hl_rules, output_rule


def _create_brain_factories(args, opts_up, opts_down, scheme):
    rounds = args.num_hidden_layers - 1

    if args.model == 'rnn':
        # Graph RNN from paper.

        def brain_up_fact():
            return LocalNet(args.n_up, args.m_up, args.hidden_width, args.conn_prob,
                            args.proj_cap, rounds, options=opts_up, update_scheme=scheme)

        def brain_down_fact():
            return LocalNet(args.n_down, args.m_down, args.hidden_width, args.conn_prob,
                            args.proj_cap, rounds, options=opts_down, update_scheme=scheme)

    else:
        # Feed-forward neural networks.

        if 'table_' in args.model:
            # Feed-forward neural networks with table-based plasticity rules.
            def rule_fact(): return _create_table_rules(args)

        elif 'reg_' in args.model:
            # Feed-forward neural networks with small-ANN-based plasticity rules.
            def rule_fact(): return _create_regression_rules(args)

        else:
            raise ValueError('Unknown model / rule type:', args.model)

        def brain_up_fact():
            hl_rules, output_rule = rule_fact()
            return FFLocalNet(
                args.n_up, args.m_up, args.num_hidden_layers, args.hidden_width,
                args.conn_prob, args.proj_cap, hl_rules=hl_rules, output_rule=output_rule,
                options=opts_up, update_scheme=scheme)

        def brain_down_fact():
            hl_rules, output_rule = rule_fact()
            return FFLocalNet(
                args.n_down, args.m_down, args.num_hidden_layers, args.hidden_width,
                args.conn_prob, args.proj_cap, hl_rules=hl_rules, output_rule=output_rule,
                options=opts_down, update_scheme=scheme)

    return brain_up_fact, brain_down_fact


def main(args):

    # Correct invalid or irrelevant parameters.
    args.model = args.model.lower()
    if args.num_hidden_layers == 1 and args.use_graph_rule:
        print('===> WARNING: Forcing use_graph_rule to False because num_hidden_layers is 1!')
        args.use_graph_rule = False
    if args.model == 'rnn' and args.downstream_backprop:
        print('===> WARNING: Forcing downstream_backprop to False because model is rnn!')
        args.downstream_backprop = False
    if args.dataset_up == 'mnist':
        args.n_up = 28 * 28
    if args.dataset_down == 'mnist':
        args.n_down = 28 * 28
    if args.vanilla:
        print('===> NOTE: Vanilla is enabled, so we do not care about rules, '
              'and only the FF versus RNN distinction matters.')
        args.downstream_backprop = False
        if args.model != 'rnn':
            print('===> Renaming model parameter to ff because it is not rnn.')
            args.model = 'ff'

    # Construct experiment tag for results file name.
    exp_tag = args.model
    if not args.vanilla:
        if args.use_graph_rule:
            exp_tag += '_gr'
        if args.use_output_rule:
            exp_tag += '_or'
    exp_tag += f'_nhl{args.num_hidden_layers}'
    exp_tag += f'_hw{args.hidden_width}'
    exp_tag += f'_cp{args.conn_prob:.1f}'
    exp_tag += f'_pc{args.proj_cap}'
    if args.universal:
        exp_tag += '_uni'
    if not args.vanilla:
        exp_tag += f'_dup{args.dataset_up}'
        exp_tag += f'_nup{args.n_up}'
        exp_tag += f'_mup{args.m_up}'
        exp_tag += f'_nre{args.num_rule_epochs}'
        exp_tag += f'_neu{args.num_epochs_upstream}'
    exp_tag += f'_ddo{args.dataset_down}'
    exp_tag += f'_ndo{args.n_down}'
    exp_tag += f'_mdo{args.m_down}'
    exp_tag += f'_ds{args.data_size}'
    exp_tag += f'_runs{args.num_runs}'
    exp_tag += f'_ned{args.num_epochs_downstream}'
    exp_tag += f'_bs{args.batch_size}'
    exp_tag += f'_lr{args.learn_rate:.3f}'
    if args.vanilla:
        exp_tag += f'_van'
    if args.downstream_backprop:
        exp_tag += f'_dsbp'
    print('Experiment tag:', exp_tag)

    # Get destination file path for results.
    dst_path = results_filepath(exp_tag + '.p')
    if os.path.isfile(dst_path) and args.ignore_if_exist:
        print('===> Already exists! Skipping...')
        sys.exit(0)

    # Get meta-learning and training options.
    opts_up = Options(gd_input=True,
                      use_graph_rule=args.use_graph_rule,
                      gd_graph_rule=args.use_graph_rule,
                      use_output_rule=args.use_output_rule,
                      gd_output_rule=args.use_output_rule,
                      gd_output=False)
    opts_down = Options(gd_input=True,
                        use_graph_rule=args.use_graph_rule,
                        gd_graph_rule=False,  # Not meta-trainable anymore!
                        use_output_rule=args.use_output_rule,
                        gd_output_rule=False,  # Not meta-trainable anymore!
                        gd_output=False)

    # Get weight update scheme.
    # Deviates from paper (works for FF but not for RNN).
    scheme_ff = UpdateScheme(cross_entropy_loss=True,
                             mse_loss=False,
                             update_misclassified_only=False,
                             update_all_edges=True)
    # Same as paper (works for RNN but not for FF).
    scheme_rnn = UpdateScheme(cross_entropy_loss=True,
                              mse_loss=False,
                              update_misclassified_only=True,
                              update_all_edges=False)
    if args.model == 'rnn':
        print('Model is RNN so selected update scheme will have:')
        print('update_misclassified_only = True')
        print('update_all_edges = False')
        scheme = scheme_rnn
    else:
        print('Model is FF so selected update scheme will have:')
        print('update_misclassified_only = False')
        print('update_all_edges = True')
        scheme = scheme_ff

    if not args.vanilla:

        # Instantiate brain factories.
        brain_fact_up, brain_fact_down = _create_brain_factories(args, opts_up, opts_down, scheme)
        min_upstream_acc = 0.4 if args.model in ['table_postcount', 'reg_onepostall'] else 0.7

        # Start evaluation.
        stats_up, stats_down = evaluate_up_down(
            brain_fact_up, brain_fact_down, args.n_up, args.n_down,
            dataset_up=args.dataset_up, dataset_down=args.dataset_down,
            downstream_backprop=args.downstream_backprop,
            num_runs=args.num_runs, num_rule_epochs=args.num_rule_epochs,
            num_epochs_upstream=args.num_epochs_upstream,
            num_epochs_downstream=args.num_epochs_downstream,
            min_upstream_acc=min_upstream_acc,
            batch_size=args.batch_size, learn_rate=args.learn_rate,
            data_size=args.data_size, relu_k=1000)

    else:

        # Instantiate brain directly.
        if args.model == 'rnn':
            def brain_fact(): return BrainNet(
                n=args.n_down, m=args.m_down, num_v=args.hidden_width, cap=args.proj_cap,
                p=args.conn_prob, rounds=args.num_hidden_layers-1, full_gd=True)
        elif args.model == 'ff':
            def brain_fact(): return FFBrainNet(
                n=args.n_down, m=args.m_down, l=args.num_hidden_layers, w=args.hidden_width,
                p=args.conn_prob, cap=args.proj_cap, full_gd=True)
        else:
            raise ValueError('Unknown model type:', args.model)
        multi_stats = evaluate_vanilla(
            brain_fact, args.n_down, dataset=args.dataset_down,
            num_runs=args.num_runs, num_epochs=args.num_epochs_downstream,
            batch_size=args.batch_size, learn_rate=args.learn_rate,
            data_size=args.data_size, relu_k=1000)
        stats_up, stats_down = None, multi_stats

    # Store all stats.
    with open(dst_path, 'wb') as f:
        pickle.dump((stats_up, stats_down), f)
    print('Stored all stats to:', dst_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
