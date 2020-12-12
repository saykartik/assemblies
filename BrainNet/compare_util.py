"""
Utilities for running the plasticity rule comparison experiments
Created by Brett Karopczyc, November 2020
"""

# Imports
import torch.nn as nn
import pickle
from eval_util import *

# ----------------------------------------------------------------------------------------------------------------------

def compare_rules(dataset='halfspace', dim=10, N=10000,
                  m=2, w=32, p=0.5, cap=16, plas_rules='output',
                  num_rule_epochs=10, num_epochs_upstream=1,
                  num_retrain=1, num_epochs_downstream=1,
                  scheme=UpdateScheme(), rules_to_skip=()):
    """For the given parameters, compare all forms of plasticity rules (along with regular GD and RNN)
    by meta-learning a rule on a small feed-forward network, and then re-training the same network
    using ONLY the learned plasticity rule. Writes all stats collected during the trainings to a pickle file."""

    # Generate training and testing datasets of equal size
    print(f'Generating {dataset} data...')
    X_train, y_train, X_test, y_test = quick_get_data(dataset, dim, N=2*N, split=0.5, relu_k=1000)

    # If we're given back too much training data (e.g. MNIST), just cap it to N
    if len(X_train) > N:
        X_train = X_train[:N, :]
        y_train = y_train[:N]

    # Define the plasticity rules we're going to compare
    low_acc = 0.8/m
    rules = {'PrePost': {'class': TableRule_PrePost, 'desc': "Table | Pre and Post"},
             'PrePostCount': {'class': TableRule_PrePostCount, 'desc': "Table | Pre, Post, and Incoming Count"},
             'PrePostPercent': {'class': TableRule_PrePostPercent, 'desc': "Table | Pre, Post and Binned Incoming Fraction"},
             'PostCount': {'class': TableRule_PostCount, 'desc': "Table | Post and Incoming Count", 'min_acc': low_acc},
             'ANNPrePost': {'class': OneBetaANNRule_PrePost, 'desc': "One Beta ANN | Pre and Post"},
             'ANNPrePostAll': {'class': OneBetaANNRule_PrePostAll, 'desc': "One Beta ANN | Pre, Post, and All Incoming"},
             'ANNOnePostAll': {'class': OneBetaANNRule_PostAll, 'desc': "One Beta ANN | Post and All Incoming", 'min_acc': low_acc},
             'ANNAllPostAll': {'class': AllBetasANNRule_PostAll, 'desc': "All Betas ANN | Post and All Incoming"}}

    print(f'Analyzing {plas_rules} plasticity rule(s)')
    use_output_rule = plas_rules is not 'hidden-layer'
    use_hl_rule = plas_rules is not 'output'

    # Define options for 'upstream' (meta) learning
    opts = Options(gd_input=True,
                   use_graph_rule=use_hl_rule,
                   gd_graph_rule=use_hl_rule,
                   use_output_rule=use_output_rule,
                   gd_output_rule=use_output_rule)

    # Use 2 hidden layers if we're learning a hidden-layer rule, otherwise just use a single hidden layer
    l = 2 if use_hl_rule else 1
    n = dim

    # Prepare to gather all stats for our experiments
    stats = {}

    # Generate a template model for feed-forward plasticity-based learning
    template = FFLocalNet(
        n, m, l, w, p, cap,
        hl_rules=TableRule_PrePost() if use_hl_rule else None,
        output_rule=TableRule_PrePost() if use_output_rule else None,
        options=opts, update_scheme=scheme)

    # Train via Gradient Descent as a baseline
    if 'GD' not in rules_to_skip:
        print('\n==== Gradient Descent Baseline ====')
        # Train an FFBrainNet with the same structure using full Gradient Descent
        gd_net = FFBrainNet(n, m, l, w, p, cap, full_gd=True)
        gd_net.copy_graph(template, True, True, True)
        train_downstream(
            X_train, y_train, gd_net, num_epochs=num_epochs_downstream,
            batch_size=100, vanilla=True, learn_rate=1e-2,
            X_test=X_test, y_test=y_test, verbose=False,
            stats_interval=100, disable_backprop=False)

        # Create a new network to re-learn just the weights normally learned by plasticity rules
        # This will give us a fair comparison to the plasticity-based re-trainings we'll perform below
        gd_net_2 = FFBrainNet(n, m, l, w, p, cap, full_gd=True)
        gd_net_2.copy_graph(template, True, True, True)

        # Copy over and freeze the weights we're NOT learning via plasticity rules
        gd_net_2.input_weights = nn.Parameter(gd_net.input_weights.clone().detach(), requires_grad=False)
        if not use_output_rule:
            gd_net_2.output_weights = nn.Parameter(gd_net.output_weights.clone().detach(), requires_grad=False)

        print()

        # Retrain only the weights that will be learned via plasticity rules for a fair comparison
        stats_down = train_downstream(
            X_train, y_train, gd_net_2, num_epochs=num_epochs_downstream,
            batch_size=100, vanilla=True, learn_rate=1e-2,
            X_test=X_test, y_test=y_test, verbose=False,
            stats_interval=100, disable_backprop=False)
        stats['GD'] = (None, stats_down)

    # Train RNN
    if 'RNN' not in rules_to_skip:
        print('\n==== Interpretation: RNN ====')
        # Train a similarly sized RNN network with the plasticity rules from the original paper
        def net_fact(): return LocalNet(n, m, w, p, cap, l-1, options=opts, update_scheme=scheme)
        model_stats = eval_rule(net_fact, None,
                                dataset, X_train, y_train, X_test, y_test,
                                num_rule_epochs, num_epochs_upstream,
                                num_retrain, num_epochs_downstream)
        stats['RNN'] = model_stats

    # For each form of plasticity rule we want to try...
    for tag, rule_dict in rules.items():
        # Skip any rules we don't want to process
        if tag in rules_to_skip:
            continue

        rule_class = rule_dict['class']
        description = rule_dict['desc']
        min_acc = rule_dict.get('min_acc', 0.7)

        print(f'\n==== Interpretation: {description} ====')

        # Meta-learn this plasticity rule on a new network
        def net_fact(): return FFLocalNet(n, m, l, w, p, cap,
                                          hl_rules=rule_class() if use_hl_rule else None,
                                          output_rule=rule_class() if use_output_rule else None,
                                          options=opts, update_scheme=scheme)
        model_stats = eval_rule(net_fact, template,
                                dataset, X_train, y_train, X_test, y_test,
                                num_rule_epochs, num_epochs_upstream,
                                num_retrain, num_epochs_downstream, min_acc)

        # Store the resulting stats, or note that the learning failed
        if model_stats == (None, None):
            print(f'Meta-learning FAILED for {description}')
        else:
            stats[tag] = model_stats

    # Save our stats to a file
    filename = f'comparing_{dataset}_{plas_rules}.p'
    dst_path = results_filepath(filename)
    with open(dst_path, 'wb') as f:
        pickle.dump(stats, f)
    print('Stored all stats to:', dst_path)


def eval_rule(net_fact, template,
              dataset, X_train, y_train, X_test, y_test,
              num_rule_epochs, num_epochs_upstream,
              num_retrain, num_epochs_downstream, min_acc=0.7):
    """Evaluate a model by meta-learning a plasticity rule, and then re-training the model using ONLY
    plasticity-based learning (without any Gradient Descent)"""

    # Meta-learn a plasticity rule(s)
    # If the resulting accuracy is too low, we may just have a bad random initialization, so try again.
    attempts = 3
    while attempts > 0:
        print(f'Meta-learning on {dataset}...')

        # Instantiate the network
        network = net_fact()

        # Copy the graph structure from the template if provided
        if template:
            network.copy_graph(template, True, True, True)

        # Meta-learn the rule
        stats_up = metalearn_rules(
            X_train, y_train, network, num_rule_epochs=num_rule_epochs,
            num_epochs=num_epochs_upstream, batch_size=100, learn_rate=1e-2,
            X_test=X_test, y_test=y_test, verbose=False)

        # Check whether the learning was successful
        success = (stats_up[2][-1] >= min_acc)
        if success:
            break
        else:
            print(f'Final upstream test acc {stats_up[2][-1]:.4f} not high enough...')
            attempts -= 1
            if attempts == 0:
                return None, None

    # Perform re-trainings on the same network and data to assess the plasticity rule we learned
    multi_stats_down = []

    for i in range(num_retrain):
        print(f'\nRe-training {i+1} of {num_retrain}:')
        # Retrain the model using the learned plasticity rule(s)
        stats_down = train_downstream(
            X_train, y_train, network, num_epochs=num_epochs_downstream,
            batch_size=100, vanilla=False, learn_rate=1e-2,
            X_test=X_test, y_test=y_test, verbose=False,
            stats_interval=100, disable_backprop=True)

        multi_stats_down.append(stats_down)

    # Return all training stats
    agg_stats_up = convert_multi_stats_uncertainty([stats_up])
    agg_stats_down = convert_multi_stats_uncertainty(multi_stats_down)

    return agg_stats_up, agg_stats_down


def plot_compare_results(stats_filename, rules_to_plot, title_desc, plot_filename):
    # Load the requests stats data
    stats_file = results_filepath(stats_filename)
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)

        # Gather the stats for each rule we want to plot
        all_stats_up = []
        all_stats_down = []
        rules_shown = []

        for rule in rules_to_plot:
            if rule in stats:
                all_stats_up.append(stats[rule][0])     # Upstream stats
                all_stats_down.append(stats[rule][1])   # Downstream stats
                rules_shown.append(rule)
            else:
                print(f"WARNING: stats for rule '{rule}' did not exist in the data!")

        if rules_shown:
            # Plot the results
            plot_compare_models(all_stats_up, all_stats_down, rules_shown,
                                f'Meta-learning on {title_desc}',
                                f'Re-training on {title_desc}',
                                plots_filepath(plot_filename))
        else:
            print('WARNING: No data to plot!')
