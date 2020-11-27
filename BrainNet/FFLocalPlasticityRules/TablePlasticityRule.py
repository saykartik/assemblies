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

from .PlasticityRule import PlasticityRule

# ----------------------------------------------------------------------------------------------------------------------

class TablePlasticityRule(PlasticityRule):
    """
    This class extends PlasticityRule to add support for rules represented as tables of beta values.
    It also enables learning of these plasticity rules via gradient descent. When specified, the plasticity rules are
    treated as PyTorch Parameters, and so have autograd enabled and can be optimized.
    """

    def initialize(self, layers=None):
        # Create the plasticity rule, stored as a flattened array of beta values
        rule_len = int(np.prod(self.rule_shape()))
        self.rule = torch.zeros(rule_len) if self.isOutputRule else torch.randn(rule_len)

        # If necessary, convert the plasticity rule to a PyTorch 'Parameter' so it's treated as a model parameter and
        # updated via an optimizer.step()
        opts = self.ff_net.options
        rule_is_learnable = opts.gd_output_rule if self.isOutputRule else opts.gd_graph_rule
        if rule_is_learnable:
            # Convert the Tensor to a Parameter
            self.rule = nn.Parameter(self.rule)

            # Register this Parameter with our parent network
            param_name = 'output_rule_param' if self.isOutputRule else f"hl_rule_param_{','.join(str(l) for l in layers)}"
            self.ff_net.register_parameter(param_name, self.rule)


    def get_rule(self):
        """Return the plasticity rule in its native shape"""
        return self.rule.clone().detach().view(self.rule_shape())

    def set_rule(self, rule):
        """Use a provided table of beta values as the rule contents"""
        # Make sure we're not trying to set a learnable rule, which currently isn't supported
        opts = self.ff_net.options
        rule_is_learnable = opts.gd_output_rule if self.isOutputRule else opts.gd_graph_rule
        assert not rule_is_learnable, "Currently, there is no support for setting learnable plasticity rules. Update the network's options as necessary."

        # Assign the rule as a flat Tensor
        if isinstance(rule, torch.Tensor):
            self.rule = rule.clone().detach().flatten().double()
        else:
            self.rule = torch.tensor(rule).flatten().double()

    def hidden_layer_betas(self, h):
        """Return the plasticity beta values for updating the weight matrix between hidden layers h and h-1"""

        # Get an index array for each dimension of the plasticity rule (corresponding to the weight matrix)
        index_arrays = self.hidden_layer_rule_index_arrays(h)

        # Convert these dimension indexes into scalar indexes into the flattened hidden-layer rule
        rule_idx = np.ravel_multi_index(index_arrays, self.rule_shape())

        # Return the corresponding array of beta values
        return self.rule[torch.tensor(rule_idx)]

    def output_betas(self, prediction, label):
        """Return the plasticity beta values for updating the weight matrix between the last hidden layer and the output layer"""

        # Get an index array for each dimension of the plasticity rule (corresponding to the weight matrix)
        index_arrays = self.output_rule_index_arrays(prediction, label)

        # Convert these dimension indexes into scalar indexes into the flattened output rule
        rule_idx = np.ravel_multi_index(index_arrays, self.rule_shape())

        # Return the corresponding array of beta values
        return self.rule[torch.tensor(rule_idx)]

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------  Methods to be implemented by subclasses  -----------------------------------------

    def rule_shape(self):
        """
        Return a tuple indicating the 'native' shape of the plasticity rule.
        This is determined by what the rule is a function of (e.g. presynaptic neuron fired?, Count of incoming neurons that fired, etc.)
        The shape is used when retrieving the rule, but the rule is stored internally as a flattened array.
        """
        # Subclasses must implement
        raise NotImplementedError()

    def hidden_layer_rule_index_arrays(self, h):
        """
        Return a tuple of index arrays (one per dimension of the rule) for each pair of nodes in hidden-layers h and h-1.
        Each array returned should:
            - have shape w[h] X w[h-1]
            - be of type 'long'
            - contain each synapse's index for the specific rule dimension
            - entry (j,i) is the value for the synapse between postsynaptic neuron j and presynaptic neuron i
        NOTE: It is safe to return entries for non-existent synapses - they will be filtered out later.
        """
        # Subclasses must implement
        raise NotImplementedError()

    def output_rule_index_arrays(self, prediction, label):
        """
        Return a tuple of index arrays (one per dimension of the rule) for each pair of nodes in the last hidden layer and the output layer.
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
