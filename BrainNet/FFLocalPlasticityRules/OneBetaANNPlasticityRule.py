# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
from .ANNPlasticityRule import ANNPlasticityRule

# ----------------------------------------------------------------------------------------------------------------------

class OneBetaANNPlasticityRule(ANNPlasticityRule):
    """
    This class extends ANNPlasticityRule to implement plasticity-rule ANNs which return a single (per-synapse) Beta value.
    i.e. The ANNs output layer has width 1
    """

    def hidden_layer_betas(self, h):
        """Return the plasticity beta values for updating the weight matrix between hidden layers h and h-1"""
        # Get the feature matrices for each input feature to the ANN
        feature_arrays = self.hidden_layer_rule_feature_arrays(h)

        # We're expecting one array per input feature
        assert len(feature_arrays) == self.rule_size()[0]

        # Form the input matrix to the plasticity rule ANN to determine all of the Betas at once:
        #   One row per synapse, columns are the input features
        input_matrix_cols = [feat.flatten() for feat in feature_arrays]
        input_matrix = torch.stack(input_matrix_cols).T

        # Call our plasticity rule ANN to determine all Beta values
        betas = self.rule(input_matrix)

        # Reshape like a weight matrix
        return betas.reshape(self.ff_net.w[h], self.ff_net.w[h-1])


    def output_betas(self, prediction, label):
        """Return the plasticity beta values for updating the weight matrix between the last hidden layer and the output layer"""
        # Get the feature matrices for each input feature to the ANN
        feature_arrays = self.output_rule_feature_arrays(prediction, label)

        # We're expecting one array per input feature
        assert len(feature_arrays) == self.rule_size()[0]

        # Form the input matrix to the plasticity rule ANN to determine all of the Betas at once:
        #   One row per synapse, columns are the input features
        input_matrix_cols = [feat.flatten() for feat in feature_arrays]
        input_matrix = torch.stack(input_matrix_cols).T

        # Call our plasticity rule ANN to determine all Beta values
        betas = self.rule(input_matrix)

        # Reshape like a weight matrix
        return betas.reshape(self.ff_net.m, self.ff_net.w[-1])


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------  Methods to be implemented by subclasses  -----------------------------------------

    def hidden_layer_rule_feature_arrays(self, h):
        """
        Return a list of feature arrays (one per input feature of the rule ANN) for each pair of nodes in hidden-layers h and h-1.
        Each array returned should:
            - have shape w[h] X w[h-1]
            - contain each synapse's value for the specific input feature
            - entry (j,i) is the value for the synapse between postsynaptic neuron j and presynaptic neuron i
        NOTE: It is safe to return entries for non-existent synapses - they will be filtered out later.
        """
        # Subclasses must implement
        raise NotImplementedError()

    def output_rule_feature_arrays(self, prediction, label):
        """
        Return a list of feature arrays (one per input feature of the rule ANN) for each pair of nodes in the last hidden layer and the output layer.
        Each array returned should:
            - have shape m X w[-1]
            - contain each synapse's value for the specific input feature
            - entry (j,i) is the value for the synapse between postsynaptic neuron j and presynaptic neuron i
        NOTE: It is safe to return entries for non-existent synapses - they will be filtered out later.

        :param prediction: The predicted label of the sample just processed
        :param label: The true label of the sample just processed
        """
        # Subclasses must implement
        raise NotImplementedError()

# ----------------------------------------------------------------------------------------------------------------------
