# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
from .FFLocalTableNet import FFLocalTableNet

# ----------------------------------------------------------------------------------------------------------------------

class FFLocalTable_PostCount(FFLocalTableNet):
    """
    This class implements the following table-based plasticity rules:

    The hidden-layer rule used to update a synapse between nodes i and j is a table of size 2 * (cap + 1):
        <node j fired?> * <number of incoming nodes to j that fired>

    The output rule used to update a synapse between node i and label j is a table of size 2 * (cap + 1):
        <node j is the sample's label?> * <number of incoming nodes to j that fired>

    As a result, we currently assume that all hidden layers have the same cap (which determines the size of the common rule)
    """

    def hidden_layer_rule_shape(self):
        return 2, self.cap[0]+1      # Postsyn fired?, Count of incoming nodes that fired

    def output_rule_shape(self):
        return 2, self.cap[0]+1      # Postsyn is the Label node?, Count of incoming nodes that fired

    def hidden_layer_rule_index_arrays(self, i):
        """
        Return index arrays for each dimension of the hidden-layer plasticity rule for the weight matrix between layers i and i-1
        """
        # Get details of the presynaptic and postsynaptic layers, their connectivity, and their latest firing patterns
        presyn_width = self.w[i - 1]
        presyn_acts = self.hidden_layer_activations[i - 1]
        postsyn_acts = self.hidden_layer_activations[i]
        connectivity = self.hidden_layers[i]

        # Rule dimension 0: 1 if the postsynaptic neuron fired, 0 otherwise
        dim0_idx = postsyn_acts.view(-1, 1).repeat(1, presyn_width).long()  # Repeat as cols for each presynaptic neuron

        # Rule dimension 1: count of incoming neurons that fired per postsynaptic neuron
        incoming_firings = presyn_acts * connectivity  # Broadcasts across rows of connectivity, yielding activity for each postsynaptic neuron
        incoming_firings = incoming_firings.sum(dim=1, keepdim=True)  # Count the number of incoming nodes that fired
        dim1_idx = incoming_firings.repeat(1, presyn_width).long()  # Repeat as cols for each presynaptic neuron

        # Return the index arrays
        return dim0_idx, dim1_idx

    def output_rule_index_arrays(self, prediction, label):
        """
        Return index arrays for each dimension of the output plasticity rule for the weight matrix between the last hidden layer and the output layer
        """
        # Get details of the presynaptic (last hidden) and postsynaptic (output) layers, and the latest firing pattern of the last hidden layer
        presyn_width = self.w[-1]
        postsyn_width = self.m
        presyn_acts = self.hidden_layer_activations[-1]
        connectivity = self.output_layer

        # Rule dimension 0: 1 if the postsynaptic node is the label, 0 otherwise
        dim0_idx = torch.zeros(postsyn_width, presyn_width, dtype=torch.long)
        dim0_idx[label] = 1

        # Rule dimension 1: count of incoming neurons that fired per postsynaptic neuron
        incoming_firings = presyn_acts * connectivity  # Broadcasts across rows of output_layer
        incoming_firings = incoming_firings.sum(dim=1, keepdim=True)    # Count the number of incoming nodes that fired per label
        dim1_idx = incoming_firings.repeat(1, presyn_width).long()  # Repeat as cols for each presynaptic neuron

        # Return the index arrays
        return dim0_idx, dim1_idx

# ----------------------------------------------------------------------------------------------------------------------
