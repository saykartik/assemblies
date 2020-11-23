# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
from .TablePlasticityRule import TablePlasticityRule

# ----------------------------------------------------------------------------------------------------------------------

class TableRule_PrePost(TablePlasticityRule):
    """
    This class implements the following table-based plasticity rule:

    The rule used to update a synapse between hidden-layer nodes i and j is a table of size 2 * 2:
        <node i fired?> * <node j fired?> (i.e. traditional Hebbian plasticity)

    The rule used to update a synapse between hidden-layer node i and output label j is a table of size 2 * 2:
        <node i fired?> * <node j is the sample's label?>
    """

    def rule_shape(self):
        return 2, 2  # Presyn fired?, Postsyn fired/Is label node?

    def hidden_layer_rule_index_arrays(self, h):
        """
        Return index arrays for each dimension of the plasticity rule for the weight matrix between hidden layers h and h-1
        """
        # Get details of the presynaptic and postsynaptic layers, their connectivity, and their latest firing patterns
        net = self.ff_net
        presyn_width = net.w[h-1]
        postsyn_width = net.w[h]
        presyn_acts = net.hidden_layer_activations[h-1]
        postsyn_acts = net.hidden_layer_activations[h]

        # Rule dimension 0: 1 if the presynaptic neuron fired, 0 otherwise
        dim0_idx = presyn_acts.repeat(postsyn_width, 1).long()  # Repeat as rows for each postsynaptic neuron

        # Rule dimension 1: 1 if the postsynaptic neuron fired, 0 otherwise
        dim1_idx = postsyn_acts.view(-1, 1).repeat(1, presyn_width).long()  # Repeat as cols for each presynaptic neuron

        # Return the index arrays
        return dim0_idx, dim1_idx

    def output_rule_index_arrays(self, prediction, label):
        """
        Return index arrays for each dimension of the plasticity rule for the weight matrix between the last hidden layer and the output layer
        """
        # Get details of the presynaptic (last hidden) and postsynaptic (output) layers, and the latest firing pattern of the last hidden layer
        net = self.ff_net
        presyn_width = net.w[-1]
        postsyn_width = net.m
        presyn_acts = net.hidden_layer_activations[-1]

        # Rule dimension 0: 1 if the presynaptic neuron fired, 0 otherwise
        dim0_idx = presyn_acts.repeat(postsyn_width, 1).long()  # Repeat as rows for each postsynaptic neuron

        # Rule dimension 1: 1 if the postsynaptic node is the label, 0 otherwise
        dim1_idx = torch.zeros(postsyn_width, presyn_width, dtype=torch.long)
        dim1_idx[label] = 1

        # Return the index arrays
        return dim0_idx, dim1_idx

# ----------------------------------------------------------------------------------------------------------------------
