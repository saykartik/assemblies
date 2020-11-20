# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
from FFLocalNet import FFLocalNet
from LocalNetBase import Options, UpdateScheme
from GDNetworks import Regression

# ----------------------------------------------------------------------------------------------------------------------

class FFLocalModelNet(FFLocalNet):
    """
    This class extends FFLocalNet to add support for local plasticity rules represented as small neural networks.

    NOTE: This class currently implements a SINGLE hidden-layer plasticity rule that applies to ALL hidden layer weight updates.
    Further development is required to increase the number of ANN-based plasticity rules supported (e.g. one per layer).
    """
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        super().__init__(n=n,
                         m=m,
                         l=l,
                         w=w,
                         p=p,
                         cap=cap,
                         options=options,
                         update_scheme=update_scheme)

        # Create plasticity rules, stored as regression models

        # Hidden layer rule
        input_sz, hidden_layer_sz, output_sz = self.hidden_layer_rule_size()
        hidden_rule_model = Regression(input_sz, hidden_layer_sz, output_sz)

        # Initialize the rule model's params
        # NOTE: How best to initialize the Regression params may require further exploration
        # hidden_rule_model.hidden.weight.data.zero_()
        hidden_rule_model.hidden.bias.data.zero_()
        # hidden_rule_model.predict.weight.data.zero_()
        hidden_rule_model.predict.bias.data.zero_()

        # If we're not learning this plasticity rule via GD, disable autograd
        if not self.options.gd_graph_rule:
            hidden_rule_model.requires_grad_(False)

        self.hidden_layer_rule = hidden_rule_model


        # Output layer rule
        input_sz, hidden_layer_sz, output_sz = self.output_rule_size()
        output_rule_model = Regression(input_sz, hidden_layer_sz, output_sz)

        # Initialize the rule model's params
        # NOTE: How best to initialize the Regression params may require further exploration
        # output_rule_model.hidden.weight.data.zero_()
        output_rule_model.hidden.bias.data.zero_()
        # output_rule_model.predict.weight.data.zero_()
        output_rule_model.predict.bias.data.zero_()

        # If we're not learning this plasticity rule via GD, disable autograd
        if not self.options.gd_output_rule:
            output_rule_model.requires_grad_(False)

        self.output_rule = output_rule_model


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------  Methods to be implemented by subclasses  -----------------------------------------

    def hidden_layer_rule_size(self):
        """
        Return a tuple indicating the layer sizes (widths) of the hidden-layer ANN plasticity rule.
        Return value should be (input_layer_width, hidden_layer_width, output_layer_width).
        """
        # Subclasses must implement
        raise NotImplementedError()

    def output_rule_size(self):
        """
        Return a tuple indicating the layer sizes (widths) of the output layer ANN plasticity rule.
        Return value should be (input_layer_width, hidden_layer_width, output_layer_width).
        """
        # Subclasses must implement
        raise NotImplementedError()

    def hidden_layer_rule_feature_arrays(self, h):
        """
        Return a list of feature arrays (one per input feature of the hidden-layer rule ANN) for hidden-layer h.
        See documentation in the subclasses for specific expectations regarding the shape and contents of the arrays.
        """
        # Subclasses must implement
        raise NotImplementedError()

    def output_rule_feature_arrays(self, prediction, label):
        """
        Return a list of feature arrays (one per input feature of the output rule ANN) for the output layer.
        See documentation in the subclasses for specific expectations regarding the shape and contents of the arrays.

        :param prediction: The predicted label of the sample just processed
        :param label: The true label of the sample just processed
        """
        # Subclasses must implement
        raise NotImplementedError()

# ----------------------------------------------------------------------------------------------------------------------
