# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author: Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
from .FFLocalModelNet import FFLocalModelNet

# ----------------------------------------------------------------------------------------------------------------------

class FFLocalAllBetasModelNet(FFLocalModelNet):
    """
    This class extends FFLocalModelNet to implement plasticity-rule ANNs which return ALL Beta values for a postsynaptic neuron.
    i.e. The ANNs output layer has width equal to the presynaptic layer
    """

    def hidden_layer_betas(self, h):
        """Return the plasticity beta values for updating the weight matrix between hidden layers h and h-1"""
        # Get the feature matrices for each input feature to the ANN
        feature_arrays = self.hidden_layer_rule_feature_arrays(h)

        # We're expecting one array per input feature
        assert len(feature_arrays) == self.hidden_layer_rule_size()[0]

        # Form the input matrix to the plasticity rule ANN to determine all of the Betas at once:
        #   One row per postsynaptic neuron, columns are the input features
        input_matrix = torch.stack(feature_arrays).T

        # Call our plasticity rule ANN to determine all Beta values
        betas = self.hidden_layer_rule(input_matrix)

        # Return the array of Beta values
        return betas


    def output_betas(self, prediction, label):
        """Return the plasticity beta values for updating the weight matrix between the last hidden layer and the output layer"""
        # Get the feature matrices for each input feature to the ANN
        feature_arrays = self.output_rule_feature_arrays(prediction, label)

        # We're expecting one array per input feature
        assert len(feature_arrays) == self.output_rule_size()[0]

        # Form the input matrix to the plasticity rule ANN to determine all of the Betas at once:
        #   One row per postsynaptic neuron, columns are the input features
        input_matrix = torch.stack(feature_arrays).T

        # Call our plasticity rule ANN to determine all Beta values
        betas = self.output_rule(input_matrix)

        # Return the array of Beta values
        return betas


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------  Methods to be implemented by subclasses  -----------------------------------------

    def hidden_layer_rule_feature_arrays(self, h):
        """
        Return a list of feature arrays (one per input feature of the hidden-layer rule ANN) for hidden-layer h.
        Each array returned should:
            - have shape (w[h],)
            - contain each postsynaptic node's value for the specific input feature
        """
        # Subclasses must implement
        raise NotImplementedError()

    def output_rule_feature_arrays(self, prediction, label):
        """
        Return a list of feature arrays (one per input feature of the output rule ANN) for the output layer.
        Each array returned should:
            - have shape (m,)
            - contain each label's value for the specific input feature

        :param prediction: The predicted label of the sample just processed
        :param label: The true label of the sample just processed
        """
        # Subclasses must implement
        raise NotImplementedError()

# ----------------------------------------------------------------------------------------------------------------------
