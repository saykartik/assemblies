'''
Run rule comparison experiments on halfspace data
Created by Brett Karopczyc, November 2020
'''

# Imports
from compare_util import compare_rules
from LocalNetBase import UpdateScheme

# Dataset params
dataset = 'halfspace'
dim = 10  # Dimension of datasets
N = 10000  # Size of datasets

# Feed-forward brain config
m = 2  # Output layer size.
w = 32  # Width of hidden layers.
p = 0.5  # Connectivity probability.
cap = 16  # Number of nodes firing per layer.

# Training config
num_retrain = 5
num_rule_epochs = 10
num_epochs_upstream = 1
num_epochs_downstream = 1
scheme = UpdateScheme(cross_entropy_loss=True,
                      mse_loss=False,
                      update_misclassified_only=False,
                      update_all_edges=True)
rules_to_skip = []

# ----------------------------------------------------------------------------------------------------------------------

# Compare rules for this configuration
for plas_rules in ['output', 'hidden-layer']:
    compare_rules(dataset=dataset, dim=dim, N=N,
                  m=m, w=w, p=p, cap=cap, plas_rules=plas_rules,
                  num_rule_epochs=num_rule_epochs, num_epochs_upstream=num_epochs_upstream,
                  num_retrain=num_retrain, num_epochs_downstream=num_epochs_downstream,
                  scheme=scheme, rules_to_skip=rules_to_skip)
