# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
from .PlasticityRule import PlasticityRule
from GDNetworks import Regression

# ----------------------------------------------------------------------------------------------------------------------

class ANNPlasticityRule(PlasticityRule):
    """
    This class extends PlasticityRule to add support for rules represented as small artificial neural networks.
    """

    def initialize(self, layers=None):
        # Create the plasticity rule, stored as a Regression model
        input_sz, hidden_layer_sz, output_sz = self.rule_size()
        rule_model = Regression(input_sz, hidden_layer_sz, output_sz)
        self.rule = rule_model

        # Initialize the rule model's params
        # NOTE: How best to initialize the Regression params may require further exploration
        # rule_model.hidden.weight.data.zero_()
        rule_model.hidden.bias.data.zero_()
        # rule_model.predict.weight.data.zero_()
        rule_model.predict.bias.data.zero_()

        # Configure the rule depending on whether we're going to learn it via GD or not:
        # either register the Regression model as a sub-module of our parent FFLocalNet,
        # or disable autograd on it entirely.
        opts = self.ff_net.options
        rule_is_learnable = opts.gd_output_rule if self.isOutputRule else opts.gd_graph_rule
        if rule_is_learnable:
            # Register this module with our parent network
            module_name = 'output_rule_module' if self.isOutputRule else f"hl_rule_module_{','.join(str(l) for l in layers)}"
            self.ff_net.add_module(module_name, rule_model)
        else:
            # If we're not learning this plasticity rule via GD, disable autograd
            rule_model.requires_grad_(False)


    def get_rule(self):
        """Return the state of the plasticity rule"""
        return self.rule.state_dict()

    def set_rule(self, rule):
        """Use a provided state dictionary as the rule contents"""
        self.rule.load_state_dict(rule, strict=True)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------  Methods to be implemented by subclasses  -----------------------------------------

    def rule_size(self):
        """
        Return a tuple indicating the layer sizes (widths) of the ANN-based plasticity rule.
        Return value should be (input_layer_width, hidden_layer_width, output_layer_width).
        """
        # Subclasses must implement
        raise NotImplementedError()

    def hidden_layer_rule_feature_arrays(self, h):
        """
        Return a list of feature arrays (one per input feature of the rule ANN) for hidden-layer h.
        See documentation in the subclasses for specific expectations regarding the shape and contents of the arrays.
        """
        # Subclasses must implement
        raise NotImplementedError()

    def output_rule_feature_arrays(self, prediction, label):
        """
        Return a list of feature arrays (one per input feature of the rule ANN) for the output layer.
        See documentation in the subclasses for specific expectations regarding the shape and contents of the arrays.

        :param prediction: The predicted label of the sample just processed
        :param label: The true label of the sample just processed
        """
        # Subclasses must implement
        raise NotImplementedError()

# ----------------------------------------------------------------------------------------------------------------------
