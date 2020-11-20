# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import numpy as np
import torch
import torch.nn as nn

from FFLocalNet import FFLocalNet
from LocalNetBase import Options, UpdateScheme

# ----------------------------------------------------------------------------------------------------------------------

class FFLocalTableNet(FFLocalNet):
    """
    This class extends FFLocalNet to add support for local plasticity rules represented as tables of beta values.
    It also enables learning of these plasticity rules via gradient descent. When specified, the plasticity rules are
    treated as PyTorch Parameters, and so have autograd enabled and can be optimized.

    NOTE: This class currently implements a SINGLE hidden-layer plasticity rule that applies to ALL hidden layer weight updates.
    Further development is required to increase the number of table-based plasticity rules supported (e.g. one per layer).
    """
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        super().__init__(n=n,
                         m=m,
                         l=l,
                         w=w,
                         p=p,
                         cap=cap,
                         options=options,
                         update_scheme=update_scheme)

        # Create plasticity rules, stored as flattened arrays of beta values
        # Hidden layer rule
        hidden_layer_rule_len = np.prod(self.hidden_layer_rule_shape())
        self.hidden_layer_rule = torch.randn(hidden_layer_rule_len)

        # Output layer rule
        output_rule_len = np.prod(self.output_rule_shape())
        self.output_rule = torch.zeros(output_rule_len)

        # If necessary, convert plasticity rules to Torch 'Parameters' so they're treated as model params and
        # updated via an optimizer.step()
        if self.options.gd_graph_rule:
            self.hidden_layer_rule = nn.Parameter(self.hidden_layer_rule)
        if self.options.gd_output_rule:
            self.output_rule = nn.Parameter(self.output_rule)


    def get_hidden_layer_rule(self):
        """Return the hidden layer plasticity rule in its native shape"""
        return self.hidden_layer_rule.clone().detach().view(self.hidden_layer_rule_shape())

    def get_output_rule(self):
        """Return the output layer plasticity rule in its native shape"""
        return self.output_rule.clone().detach().view(self.output_rule_shape())

    def set_hidden_layer_rule(self, rule):
        self.hidden_layer_rule = torch.tensor(rule).flatten().double()

    def set_output_rule(self, rule):
        self.output_rule = rule.clone().detach().flatten().double()


    def hidden_layer_betas(self, i):
        """Return the plasticity beta values for updating the weight matrix between hidden layers i and i-1"""

        # Get an index array for each dimension of the plasticity rule (corresponding to the weight matrix)
        index_arrays = self.hidden_layer_rule_index_arrays(i)

        # Convert these dimension indexes into scalar indexes into the flattened hidden-layer rule
        rule_idx = np.ravel_multi_index(index_arrays, self.hidden_layer_rule_shape())

        # Return the corresponding array of beta values
        return self.hidden_layer_rule[torch.tensor(rule_idx)]

    def output_betas(self, prediction, label):
        """Return the plasticity beta values for updating the weight matrix between the last hidden layer and the output layer"""

        # Get an index array for each dimension of the plasticity rule (corresponding to the weight matrix)
        index_arrays = self.output_rule_index_arrays(prediction, label)

        # Convert these dimension indexes into scalar indexes into the flattened output rule
        rule_idx = np.ravel_multi_index(index_arrays, self.output_rule_shape())

        # Return the corresponding array of beta values
        return self.output_rule[torch.tensor(rule_idx)]

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------  Methods to be implemented by subclasses  -----------------------------------------

    def hidden_layer_rule_shape(self):
        """
        Return a tuple indicating the 'native' shape of the hidden-layer plasticity rule.
        This is determined by what the rule is a function of (e.g. presynaptic neuron fired?, Count of incoming neurons that fired, etc.)
        The shape is used when retrieving the rule, but the rule is stored internally as a flattened array.
        """
        # Subclasses must implement
        raise NotImplementedError()

    def output_rule_shape(self):
        """
        Return a tuple indicating the 'native' shape of the output layer plasticity rule.
        This is determined by what the rule is a function of (e.g. presynaptic neuron fired?, Count of incoming neurons that fired, etc.)
        The shape is used when retrieving the rule, but the rule is stored internally as a flattened array.
        """
        # Subclasses must implement
        raise NotImplementedError()

    def hidden_layer_rule_index_arrays(self, i):
        """
        Return a tuple of index arrays (one per dimension of the hidden-layer rule) for each pair of nodes in hidden-layers i and i-1.
        Each array returned should:
            - have shape w[i] X w[i-1]
            - be of type 'long'
            - contain each synapse's index for the specific rule dimension
            - entry (j,i) is the value for the synapse between postsynaptic neuron j and presynaptic neuron i
        NOTE: It is safe to return entries for non-existent synapses - they will be filtered out later.
        """
        # Subclasses must implement
        raise NotImplementedError()

    def output_rule_index_arrays(self, prediction, label):
        """
        Return a tuple of index arrays (one per dimension of the output rule) for each pair of nodes in the last hidden layer and the output layer.
        Each array returned should:
            - have shape m X w[-1]
            - be of type 'long'
            - contain each synapse's index for the specific rule dimension
            - entry (j,i) is the value for the synapse between postsynaptic neuron j and presynaptic neuron i
        NOTE: It is safe to return entries for non-existent synapses - they will be filtered out later.

        :param prediction: The predicted label of the sample just processed
        :param label: The true label of the sample just processed
        """
        # Subclasses must implement
        raise NotImplementedError()

# ----------------------------------------------------------------------------------------------------------------------
