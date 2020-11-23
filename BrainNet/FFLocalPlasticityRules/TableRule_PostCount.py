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

class TableRule_PostCount(TablePlasticityRule):
    """
    This class implements the following table-based plasticity rule:

    The rule used to update a synapse between hidden-layer nodes i and j is a table of size 2 * (cap + 1):
        <node j fired?> * <number of incoming nodes to j that fired>

    The rule used to update a synapse between hidden-layer node i and output label j is a table of size 2 * (cap + 1):
        <node j is the sample's label?> * <number of incoming nodes to j that fired>

    As a result, we require that all layers this rule applies to have the same cap (for the presynaptic layer).
    """

    def __init__(self):
        super().__init__()
        # Define attributes used by this class
        self.presynaptic_cap = None     # Will be assigned in initialize()

    def initialize(self, layers=None):
        # Determine the common cap of all presynaptic layers
        if self.isOutputRule:
            self.presynaptic_cap = self.ff_net.cap[-1]
        else:
            caps = {self.ff_net.cap[lay-1] for lay in layers}
            assert len(caps) == 1, "Caps of presynaptic layers were inconsistent"
            self.presynaptic_cap = caps.pop()

        # Call up to our super's initialize()
        super().initialize(layers)

    def rule_shape(self):
        return 2, self.presynaptic_cap+1      # Postsyn fired/Is Label node?, Count of incoming nodes that fired

    def hidden_layer_rule_index_arrays(self, h):
        """
        Return index arrays for each dimension of the plasticity rule for the weight matrix between hidden layers h and h-1
        """
        # Get details of the presynaptic and postsynaptic layers, their connectivity, and their latest firing patterns
        net = self.ff_net
        presyn_width = net.w[h-1]
        presyn_acts = net.hidden_layer_activations[h-1]
        postsyn_acts = net.hidden_layer_activations[h]
        connectivity = net.hidden_layers[h]

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
        Return index arrays for each dimension of the plasticity rule for the weight matrix between the last hidden layer and the output layer
        """
        # Get details of the presynaptic (last hidden) and postsynaptic (output) layers, and the latest firing pattern of the last hidden layer
        net = self.ff_net
        presyn_width = net.w[-1]
        postsyn_width = net.m
        presyn_acts = net.hidden_layer_activations[-1]
        connectivity = net.output_layer

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
