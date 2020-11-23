# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
from .AllBetasANNPlasticityRule import AllBetasANNPlasticityRule

# ----------------------------------------------------------------------------------------------------------------------

class AllBetasANNRule_PostAll(AllBetasANNPlasticityRule):
    """
    This class implements the following ANN-based plasticity rule:

    The rule used to update ALL synapses coming in to postsynaptic neuron j is an ANN with 1+<hidden-layer width> input features:
        <node j fired?>, <all firings of presynaptic layer as {1,-1,0}>
    The output is a list of Beta values, one per (possible) incoming synapse.

    The rule used to update ALL synapses coming in to output label j is an ANN with 1+<hidden-layer width> input features:
        <node j is the sample's label?>, <all firings of presynaptic layer as {1,-1,0}>
    The output is a list of Beta values, one per (possible) incoming synapse.
    """

    def __init__(self):
        super().__init__()
        # Define attributes used by this class
        self.presynaptic_width = None     # Will be assigned in initialize()

    def initialize(self, layers=None):
        # Determine the common width of all presynaptic layers
        if self.isOutputRule:
            self.presynaptic_width = self.ff_net.w[-1]
        else:
            widths = {self.ff_net.w[lay-1] for lay in layers}
            assert len(widths) == 1, "Widths of presynaptic layers were inconsistent"
            self.presynaptic_width = widths.pop()

        # Call up to our super's initialize()
        super().initialize(layers)


    def rule_size(self):
        return 1+self.presynaptic_width, 20, self.presynaptic_width  # Input size, Hidden layer size, Output size


    def hidden_layer_rule_feature_arrays(self, h):
        """Return the feature arrays for each input feature of the rule ANN for the weight matrix between hidden layers h and h-1"""
        feature_arrays = []

        # Get details of the presynaptic and postsynaptic layers, their connectivity, and their latest firing patterns
        net = self.ff_net
        presyn_width = net.w[h-1]
        postsyn_width = net.w[h]
        presyn_acts = net.hidden_layer_activations[h-1]
        postsyn_acts = net.hidden_layer_activations[h]
        connectivity = net.hidden_layers[h]

        # First feature is 1 if the postsynaptic neuron fired, 0 otherwise
        feature_arrays.append(postsyn_acts)

        # Remaining features are the activations for each presynaptic neuron:
        #    1 if the neuron fired, -1 if it didn't fire, 0 if it is not connected to the postsynaptic neuron
        # For each presynaptic neuron...
        for i in range(presyn_width):
            # Form an array for the firing of presynaptic neuron i
            # 1 if the neuron fired, -1 if it didn't fire
            presyn_fired = torch.ones(postsyn_width)
            if not presyn_acts[i]:
                presyn_fired = -presyn_fired

            # Determine whether presynaptic neuron i is connected to each of the postsynaptic neurons
            # This is the ith column of the connectivity matrix
            i_connectivity = connectivity[:, i]

            # Zero out the activations if there is no connection to the postsynaptic neuron
            presyn_fired = presyn_fired * i_connectivity  # Element-wise multiply

            # Add this feature to our list
            feature_arrays.append(presyn_fired)

        # Return the feature arrays
        return feature_arrays


    def output_rule_feature_arrays(self, prediction, label):
        """Return the feature arrays for each input feature of the output rule ANN for the weight matrix between the last hidden layer and the output layer"""
        feature_arrays = []

        # Get details of the presynaptic (last hidden) and postsynaptic (output) layers, and the latest firing pattern of the last hidden layer
        net = self.ff_net
        presyn_width = net.w[-1]
        postsyn_width = net.m
        presyn_acts = net.hidden_layer_activations[-1]

        # First feature is 1 if the postsynaptic neuron is the label, 0 otherwise
        postsyn_labels = torch.zeros(postsyn_width)
        postsyn_labels[label] = 1
        feature_arrays.append(postsyn_labels)

        # Remaining features are the activations for each presynaptic neuron:
        #    1 if the neuron fired, -1 if it didn't fire, 0 if it is not connected to the postsynaptic neuron
        # For each presynaptic neuron...
        for i in range(presyn_width):
            # Form an array for the firing of presynaptic neuron i
            # 1 if the neuron fired, -1 if it didn't fire
            presyn_fired = torch.ones(postsyn_width)
            if not presyn_acts[i]:
                presyn_fired = -presyn_fired

            # Determine whether presynaptic neuron i is connected to each of the postsynaptic neurons
            # This is the ith column of the output_layer connectivity matrix
            connectivity = net.output_layer[:, i]

            # Zero out the activations if there is no connection to the postsynaptic neuron
            presyn_fired = presyn_fired * connectivity      # Element-wise multiply

            # Add this feature to our list
            feature_arrays.append(presyn_fired)

        # Return the feature arrays
        return feature_arrays

# ----------------------------------------------------------------------------------------------------------------------
