# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
class PlasticityRule:
    """
    Instances of this class represent local plasticity rules used by FFLocalNet objects.
    PlasticityRule itself is an abstract class that defines the basic structure and interface of these objects.
    Subclasses are required to override and implement the methods defined below to provide concrete instances.
    """
    def __init__(self):
        # Define basic attributes of PlasticityRules
        # NOTE: Before overriding in subclasses, read the explanation of the initialize() method below.
        # PlasticityRules are expected to perform most of their initialization in initialize() rather than __init__().

        # The following two attributes are assigned by FFLocalNet.__init__() when a PlasticityRule object
        # is passed in either the 'hl_rules' or 'output_rule' parameters:
        self.ff_net = None          # The FFLocalNet object that uses this PlasticityRule
        self.isOutputRule = False   # True if the rule is used as an output-layer rule, False if used as a hidden-layer rule

        self.rule = None    # The content of the rule's state, as determined by subclasses (could be any type)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------  Methods to be implemented by subclasses  -----------------------------------------

    def initialize(self, layers=None):
        """
        When a PlasticityRule is passed in to the FFLocalNet constructor, its basic parameters are defined and then
        initialize() is called on the rule to perform any further initialization necessary. Subclasses are expected
        to use this opportunity to initialize any state required by the rule.

        At this point, both the ff_net and isOutputRule attributes have been set, giving the PlasticityRule access to
        its 'owning' FFLocalNet object. In addition, for PlasticityRules used to update hidden-layer weights, a list of
        hidden-layer indices the rule is applied to is passed as a parameter. This is useful if the PlasticityRule
        needs specific details of the FFLocalNet in order to properly initialize (e.g. the width of hidden-layers).

        :param layers: For PlasticityRules assigned as hidden-layer rules, a list of hidden-layer indices to which the
                       rule has been applied.
        """
        raise NotImplementedError()

    def hidden_layer_betas(self, h):
        """
        Returns a 2D array of plasticity beta values for updating the weight matrix between hidden layers h and h-1
        This array should have shape w[h] x w[h-1]
        NOTE: It is safe to return entries for non-existent synapses - they will be ignored.
        """
        # Subclasses must implement
        raise NotImplementedError()

    def output_betas(self, prediction, label):
        """
        Returns a 2D array of plasticity beta values for updating the weight matrix between the last hidden layer and the output layer
        This array should have shape m x w[-1]
        NOTE: It is safe to return entries for non-existent synapses - they will be ignored.

        :param prediction: The predicted label of the sample just processed
        :param label: The true label of the sample just processed
        """
        # Subclasses must implement
        raise NotImplementedError()

# ----------------------------------------------------------------------------------------------------------------------
