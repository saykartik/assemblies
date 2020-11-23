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

class OneBetaANNRule_PrePostAll(OneBetaANNPlasticityRule):
    """
    This class implements the following ANN-based plasticity rule:

    The rule used to update a synapse between hidden-layer nodes i and j is an ANN with 2+<hidden-layer width> input features:
        <node i fired?>, <node j fired?>, <all firings of presynaptic layer as {1,-1,0}>
    The output is a single Beta value used to update the specified synapse.

    The rule used to update a synapse between hidden-layer node i and output label j is an ANN with 2+<hidden-layer width> input features:
        <node i fired?>, <node j is the sample's label?>, <all firings of presynaptic layer as {1,-1,0}>
    The output is a single Beta value used to update the specified synapse.
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
        return 2+self.presynaptic_width, 20, 1      # Input size, Hidden layer size, Output size


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

        # First feature is 1 if the presynaptic neuron fired, 0 otherwise
        presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)  # Repeat as rows for each postsynaptic neuron
        feature_arrays.append(presyn_act_matrix)

        # Second feature is 1 if the postsynaptic neuron fired, 0 otherwise
        postsyn_act_matrix = postsyn_acts.view(-1, 1).repeat(1, presyn_width)  # Repeat as cols for each presynaptic neuron
        feature_arrays.append(postsyn_act_matrix)

        # Remaining features are the activations for each presynaptic neuron:
        #    1 if the neuron fired, -1 if it didn't fire, 0 if it is not connected to the postsynaptic neuron
        # For each presynaptic neuron...
        for i in range(presyn_width):
            # Form a matrix for the firing of presynaptic neuron i
            # 1 if the neuron fired, -1 if it didn't fire
            act_matrix = torch.ones(postsyn_width, presyn_width)
            if not presyn_acts[i]:
                act_matrix = -act_matrix

            # Determine whether presynaptic neuron i is connected to each of the postsynaptic neurons
            # This is the ith column of the connectivity matrix
            i_connectivity = connectivity[:, i:i+1]

            # Zero out the activations if there is no connection to the postsynaptic neuron
            act_matrix = act_matrix * i_connectivity  # Broadcasts across cols of a act_matrix

            # Add this feature to our list
            feature_arrays.append(act_matrix)

        # Return the feature arrays
        return feature_arrays


    def output_rule_feature_arrays(self, prediction, label):
        """Return the feature arrays for each input feature of the rule ANN for the weight matrix between the last hidden layer and the output layer"""
        feature_arrays = []

        # Get details of the presynaptic (last hidden) and postsynaptic (output) layers, and the latest firing pattern of the last hidden layer
        net = self.ff_net
        presyn_width = net.w[-1]
        postsyn_width = net.m
        presyn_acts = net.hidden_layer_activations[-1]

        # First feature is 1 if the presynaptic neuron fired, 0 otherwise
        presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)  # Repeat as rows for each postsynaptic neuron
        feature_arrays.append(presyn_act_matrix)

        # Second feature is 1 if the postsynaptic neuron is the label, 0 otherwise
        postsyn_label_matrix = torch.zeros(postsyn_width, presyn_width)
        postsyn_label_matrix[label] = 1
        feature_arrays.append(postsyn_label_matrix)

        # Remaining features are the activations for each presynaptic neuron:
        #    1 if the neuron fired, -1 if it didn't fire, 0 if it is not connected to the postsynaptic neuron
        # For each presynaptic neuron...
        for i in range(presyn_width):
            # Form a matrix for the firing of presynaptic neuron i
            # 1 if the neuron fired, -1 if it didn't fire
            act_matrix = torch.ones(postsyn_width, presyn_width)
            if not presyn_acts[i]:
                act_matrix = -act_matrix

            # Determine whether presynaptic neuron i is connected to each of the postsynaptic neurons
            # This is the ith column of the output_layer connectivity matrix
            connectivity = net.output_layer[:, i:i+1]

            # Zero out the activations if there is no connection to the postsynaptic neuron
            act_matrix = act_matrix * connectivity      # Broadcasts across cols of a act_matrix

            # Add this feature to our list
            feature_arrays.append(act_matrix)

        # Return the feature arrays
        return feature_arrays

# ----------------------------------------------------------------------------------------------------------------------
