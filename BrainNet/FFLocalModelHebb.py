# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author: Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
from FFLocalModelNet import FFLocalModelNet

# ----------------------------------------------------------------------------------------------------------------------

class FFLocalModelHebb(FFLocalModelNet):
    """
    This class implements the following ANN-based plasticity rules:

    The hidden-layer rule used to update a synapse between nodes i and j is an ANN with 2 input features:
        <node i fired?>, <node j fired?>    (i.e. traditional Hebbian plasticity)

    The output rule used to update a synapse between node i and label j is an ANN with 2 input features:
        <node i fired?>, <node j is the sample's label?>
    """

    def hidden_layer_rule_size(self):
        return 2, 2      # Input size, Hidden layer size

    def output_rule_size(self):
        return 2, 2      # Input size, Hidden layer size

    def hidden_layer_betas(self, i):
        """Return the plasticity beta values for updating the weight matrix between hidden layers i and i-1"""
        # Get details of the presynaptic and postsynaptic layers, their connectivity, and their latest firing patterns
        presyn_width = self.w[i - 1]
        postsyn_width = self.w[i]
        presyn_acts = self.hidden_layer_activations[i - 1]
        postsyn_acts = self.hidden_layer_activations[i]

        # Form the input matrix to the plasticity rule ANN to determine all of the Betas at once:
        # One row per synapse, first col is 1 if presynaptic neuron fired, second col is 1 if postsynaptic neuron fired
        presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)  # Repeat as rows for each postsynaptic neuron
        postsyn_act_matrix = postsyn_acts.view(-1, 1).repeat(1, presyn_width)  # Repeat as cols for each presynaptic neuron
        input_matrix = torch.stack((presyn_act_matrix.flatten(), postsyn_act_matrix.flatten())).T

        # Call our plasticity rule ANN to determine all Beta values
        betas = self.hidden_layer_rule(input_matrix)

        # Reshape like a weight matrix
        return betas.reshape(postsyn_width, presyn_width)

    def output_betas(self, prediction, label):
        """Return the plasticity beta values for updating the weight matrix between the last hidden layer and the output layer"""
        # Get details of the presynaptic (last hidden) and postsynaptic (output) layers, and the latest firing pattern of the last hidden layer
        presyn_width = self.w[-1]
        postsyn_width = self.m
        presyn_acts = self.hidden_layer_activations[-1]

        # Form the input matrix to the plasticity rule ANN to determine all of the Betas at once:
        # One row per synapse, first col is 1 if presynaptic neuron fired, second col is 1 if postsynaptic neuron is the label
        presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)  # Repeat as rows for each postsynaptic neuron
        postsyn_label_matrix = torch.zeros(postsyn_width, presyn_width)
        postsyn_label_matrix[label] = 1
        input_matrix = torch.stack((presyn_act_matrix.flatten(), postsyn_label_matrix.flatten())).T

        # Call our plasticity rule ANN to determine all Beta values
        betas = self.output_rule(input_matrix)

        # Reshape like a weight matrix
        return betas.reshape(postsyn_width, presyn_width)

# ----------------------------------------------------------------------------------------------------------------------
