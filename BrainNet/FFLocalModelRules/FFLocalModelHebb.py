# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author: Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
from .FFLocalOneBetaModelNet import FFLocalOneBetaModelNet

# ----------------------------------------------------------------------------------------------------------------------

class FFLocalModelHebb(FFLocalOneBetaModelNet):
    """
    This class implements the following ANN-based plasticity rules:

    The hidden-layer rule used to update a synapse between nodes i and j is an ANN with 2 input features:
        <node i fired?>, <node j fired?>    (i.e. traditional Hebbian plasticity)
    The output is a single Beta value used to update the specified synapse.

    The output rule used to update a synapse between node i and label j is an ANN with 2 input features:
        <node i fired?>, <node j is the sample's label?>
    The output is a single Beta value used to update the specified synapse.
    """

    def hidden_layer_rule_size(self):
        return 2, 2, 1      # Input size, Hidden layer size, Output size

    def output_rule_size(self):
        return 2, 2, 1      # Input size, Hidden layer size, Output size


    def hidden_layer_rule_feature_arrays(self, h):
        """Return the feature arrays for each input feature of the hidden-layer rule ANN for the weight matrix between layers h and h-1"""

        # Get details of the presynaptic and postsynaptic layers, their connectivity, and their latest firing patterns
        presyn_width = self.w[h-1]
        postsyn_width = self.w[h]
        presyn_acts = self.hidden_layer_activations[h-1]
        postsyn_acts = self.hidden_layer_activations[h]

        # First feature is 1 if the presynaptic neuron fired, 0 otherwise
        presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)  # Repeat as rows for each postsynaptic neuron

        # Second feature is 1 if the postsynaptic neuron fired, 0 otherwise
        postsyn_act_matrix = postsyn_acts.view(-1, 1).repeat(1, presyn_width)  # Repeat as cols for each presynaptic neuron

        # Return the feature arrays
        return [presyn_act_matrix, postsyn_act_matrix]


    def output_rule_feature_arrays(self, prediction, label):
        """Return the feature arrays for each input feature of the output rule ANN for the weight matrix between the last hidden layer and the output layer"""

        # Get details of the presynaptic (last hidden) and postsynaptic (output) layers, and the latest firing pattern of the last hidden layer
        presyn_width = self.w[-1]
        postsyn_width = self.m
        presyn_acts = self.hidden_layer_activations[-1]

        # First feature is 1 if the presynaptic neuron fired, 0 otherwise
        presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)  # Repeat as rows for each postsynaptic neuron

        # Second feature is 1 if the postsynaptic neuron is the label, 0 otherwise
        postsyn_label_matrix = torch.zeros(postsyn_width, presyn_width)
        postsyn_label_matrix[label] = 1

        # Return the feature arrays
        return [presyn_act_matrix, postsyn_label_matrix]

# ----------------------------------------------------------------------------------------------------------------------
