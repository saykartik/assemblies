"""
Generate plots for the rule comparison experiments
Created by Brett Karopczyc, November 2020
"""

from compare_util import *

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
