# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
from .OneBetaANNPlasticityRule import OneBetaANNPlasticityRule

# ----------------------------------------------------------------------------------------------------------------------

class OneBetaANNRule_PrePost(OneBetaANNPlasticityRule):
    """
    This class implements the following ANN-based plasticity rule:

    The rule used to update a synapse between hidden-layer nodes i and j is an ANN with 2 input features:
        <node i fired?>, <node j fired?>    (i.e. traditional Hebbian plasticity)
    The output is a single Beta value used to update the specified synapse.

    The rule used to update a synapse between hidden-layer node i and output label j is an ANN with 2 input features:
        <node i fired?>, <node j is the sample's label?>
    The output is a single Beta value used to update the specified synapse.
    """

    def rule_size(self):
        return 2, 2, 1  # Input size, Hidden layer size, Output size


    def hidden_layer_rule_feature_arrays(self, h):
        """Return the feature arrays for each input feature of the rule ANN for the weight matrix between hidden layers h and h-1"""

        # Get details of the presynaptic and postsynaptic layers, their connectivity, and their latest firing patterns
        net = self.ff_net
        presyn_width = net.w[h-1]
        postsyn_width = net.w[h]
        presyn_acts = net.hidden_layer_activations[h-1]
        postsyn_acts = net.hidden_layer_activations[h]

        # First feature is 1 if the presynaptic neuron fired, 0 otherwise
        presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)  # Repeat as rows for each postsynaptic neuron

        # Second feature is 1 if the postsynaptic neuron fired, 0 otherwise
        postsyn_act_matrix = postsyn_acts.view(-1, 1).repeat(1, presyn_width)  # Repeat as cols for each presynaptic neuron

        # Return the feature arrays
        return [presyn_act_matrix, postsyn_act_matrix]


    def output_rule_feature_arrays(self, prediction, label):
        """Return the feature arrays for each input feature of the rule ANN for the weight matrix between the last hidden layer and the output layer"""

        # Get details of the presynaptic (last hidden) and postsynaptic (output) layers, and the latest firing pattern of the last hidden layer
        net = self.ff_net
        presyn_width = net.w[-1]
        postsyn_width = net.m
        presyn_acts = net.hidden_layer_activations[-1]

        # First feature is 1 if the presynaptic neuron fired, 0 otherwise
        presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)  # Repeat as rows for each postsynaptic neuron

        # Second feature is 1 if the postsynaptic neuron is the label, 0 otherwise
        postsyn_label_matrix = torch.zeros(postsyn_width, presyn_width)
        postsyn_label_matrix[label] = 1

        # Return the feature arrays
        return [presyn_act_matrix, postsyn_label_matrix]

# ----------------------------------------------------------------------------------------------------------------------
