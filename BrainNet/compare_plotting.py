"""
Generate plots for the rule comparison experiments
Created by Brett Karopczyc, November 2020
"""

from compare_util import *

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

# Common sets of rules
baseline_rules = ['GD', 'RNN']
table_based_rules = ['PrePost', 'PrePostCount', 'PrePostPercent', 'PostCount']
ann_based_rules = ['ANNPrePost', 'ANNPrePostAll', 'ANNOnePostAll', 'ANNAllPostAll']
good_rules = ['PrePost', 'PrePostCount', 'PrePostPercent', 'ANNPrePost', 'ANNPrePostAll', 'ANNAllPostAll']
all_rules = baseline_rules + table_based_rules + ann_based_rules

# ----------------------------------------------------------------------------------------------------------------------
# Halfspace plots

# Output Rule Only
stats_file = 'comparing_halfspace_output.p'
plot_compare_results(stats_file, table_based_rules, 'Halfspace (Output rule)', 'compare_halfspace_table_output')
plot_compare_results(stats_file, ann_based_rules, 'Halfspace (Output rule)', 'compare_halfspace_ann_output')
plot_compare_results(stats_file, baseline_rules + good_rules, 'Halfspace (Output rule)', 'compare_halfspace_output')

# Hidden-Layer Rule Only
stats_file = 'comparing_halfspace_hidden-layer.p'
#plot_compare_results(stats_file, table_based_rules, 'Halfspace (Hidden-Layer rule)', 'compare_halfspace_table_hl')
#plot_compare_results(stats_file, ann_based_rules, 'Halfspace (Hidden-Layer rule)', 'compare_halfspace_ann_hl')
plot_compare_results(stats_file, baseline_rules + good_rules, 'Halfspace (Hidden-Layer rule)', 'compare_halfspace_hl')

# ----------------------------------------------------------------------------------------------------------------------
# ReLU plots

# Output Rule Only
stats_file = 'comparing_relu_output.p'
# plot_compare_results(stats_file, good_rules, 'ReLU (Output rule)', 'compare_relu_all_output')
plot_compare_results(stats_file, baseline_rules + good_rules, 'ReLU (Output rule)', 'compare_relu_output')

# Hidden-Layer Rule Only
stats_file = 'comparing_relu_hidden-layer.p'
# plot_compare_results(stats_file, good_rules, 'ReLU (Hidden-Layer rule)', 'compare_relu_all_hl')
plot_compare_results(stats_file, baseline_rules + good_rules, 'ReLU (Hidden-Layer rule)', 'compare_relu_hl')

# ----------------------------------------------------------------------------------------------------------------------
# MNIST plots

# Output Rule Only
stats_file = 'comparing_mnist_output.p'
plot_compare_results(stats_file, baseline_rules + good_rules, 'MNIST (Output rule)', 'compare_mnist_output')

# Hidden-Layer Rule Only
stats_file = 'comparing_mnist_hidden-layer.p'
plot_compare_results(stats_file, baseline_rules + good_rules, 'MNIST (Hidden-Layer rule)', 'compare_mnist_hl')

# ----------------------------------------------------------------------------------------------------------------------
