# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
import torch.nn as nn

from FFBrainNet import FFBrainNet
from FFLocalPlasticityRules.PlasticityRule import PlasticityRule
from FFLocalPlasticityRules.TablePlasticityRule import TablePlasticityRule
from LocalNetBase import Options, UpdateScheme

# ----------------------------------------------------------------------------------------------------------------------

class FFLocalNet(FFBrainNet):
    """
    This class extends FFBrainNet to add support for local plasticity rules, in a manner analogous to LocalNetBase.
    This support includes learning and using plasticity rules for both hidden layers and the output layer.

    An instance of FFLocalNet shares the same network structure as FFBrainNet:
        <input layer> [any number of hidden layers] <output layer>, with random inter-layer connectivity

    The basic parameterization of the class is the same as FFBrainNet:
        n = number of input features (i.e. nodes in the input layer)
        m = number of output labels (i.e. nodes in the output layer)
        l = number of hidden layers (between input and output layers)
        w = common width (# of nodes) of all hidden layers OR array of length l indicating the width of each hidden layer
        p = inter-layer connectivity probability OR array of length l indicating connectivity probability between each
            hidden layer and the preceding layer. NOTE: The output layer is *fully-connected* with the last hidden layer.
        cap = max number of nodes firing at the hidden layers OR array of length l indicating the cap per hidden layer
        hl_rules = common PlasticityRule to use for all hidden layers OR array of length (l-1) containing the
                   PlasticityRules to use for updating the weight matrices between the l hidden layers.
                   NOTE: The same PlasticityRule object can be included multiple times in the list to share the same
                   rule for multiple layers. Must be supplied when use_graph_rule=True.
        output_rule = the PlasticityRule to use for the output layer. Must be supplied when use_output_rule=True.

    In addition, the Options and UpdateScheme structures from LocalNetBase were reused to control how the class is trained.
    """
    def __init__(self, n, m, l, w, p, cap, hl_rules=None, output_rule=None, options=Options(), update_scheme=UpdateScheme()):
        super().__init__(n=n,
                         m=m,
                         l=l,
                         w=w,
                         p=p,
                         cap=cap,
                         gd_input=options.gd_input,
                         gd_output=options.gd_output)

        # Make sure the options are consistent
        assert not options.use_input_rule, "There is currently no support for an input layer plasticity rule"
        assert options.gd_input, "If we don't use GD on the input weights, they will never be learned"

        assert options.use_graph_rule == (l > 1), "A graph rule should be used iff there is more than 1 hidden layer"
        assert options.use_graph_rule or not options.gd_graph_rule, "gd_graph_rule is not applicable when use_graph_rule is False"

        assert options.use_output_rule or not options.gd_output_rule, "gd_output_rule is not applicable when use_output_rule is False"
        assert options.use_output_rule != options.gd_output, "use_output_rule and gd_output should be mutually exclusive"

        # Store additional params
        self.options = options
        self.update_scheme = update_scheme
        self.step_sz = 0.01

        # Define our plasticity rules:

        # Hidden Layer rules
        # Make sure a hidden-layer rule was supplied if needed
        hl_rules_supplied = bool(hl_rules)
        hl_rule_needed = options.use_graph_rule and (l > 1)
        assert hl_rules_supplied == hl_rule_needed, "The hl_rules parameter does not agree with the other parameters"

        # Convert to a list if a single rule was supplied
        if not hl_rules_supplied:
            hl_rules = [None]   # First hidden layer never uses a plasticity rule
        elif isinstance(hl_rules, PlasticityRule):
            hl_rules = [None] + [hl_rules] * (l-1)     # Common plasticity rule for all hidden layers
        else:
            assert len(hl_rules) == (l-1), "hl_rules list must have length (l-1)"
            hl_rules = [None] + hl_rules

        # Initialize the rules
        unique_rules = set(hl_rules) - {None}
        for rule in unique_rules:
            # Assign basic params
            rule.ff_net = self
            rule.isOutputRule = False

            # Ask the rule to initialize itself
            layers = [i for i, r in enumerate(hl_rules) if r == rule]
            rule.initialize(layers=layers)

        # Store these rules
        self.hidden_layer_rules = hl_rules


        # Output rule
        # Make sure an output rule was supplied if needed
        output_rule_supplied = bool(output_rule)
        assert output_rule_supplied == options.use_output_rule, "The output_rule parameter does not agree with options.use_output_rule"

        # Initialize the rule
        if output_rule_supplied:
            # Make sure the output rule is distinct from all hidden-layer rules
            assert output_rule not in hl_rules, "The output rule must be distinct from all hidden-layer rules"

            # Assign basic params
            output_rule.ff_net = self
            output_rule.isOutputRule = True

            # Ask the rule to initialize itself
            output_rule.initialize()
        else:
            output_rule = None

        # Store the rule
        self.output_rule = output_rule


    def get_hidden_layer_rule(self):
        """Return the hidden layer plasticity rule(s)"""
        unique_rules = set(self.hidden_layer_rules) - {None}

        # If the only hidden layer rule is a table-based rule, return the rule in its native shape
        if len(unique_rules) == 1:
            only_rule = unique_rules.pop()
            if isinstance(only_rule, TablePlasticityRule):
                return only_rule.get_rule()
            else:
                return only_rule
        else:
            return unique_rules

    def get_output_rule(self):
        """Return the output layer plasticity rule"""
        if isinstance(self.output_rule, TablePlasticityRule):
            return self.output_rule.get_rule()
        else:
            return self.output_rule

    def set_hidden_layer_rule(self, rule):
        unique_rules = set(self.hidden_layer_rules) - {None}
        if len(unique_rules) == 1:
            only_rule = unique_rules.pop()
            if isinstance(only_rule, TablePlasticityRule):
                # We can set this rule
                only_rule.set_rule(rule)
                return
        raise AssertionError('Currently, set_hidden_layer_rule() can only be called for single, table-based plasticity rules.')

    def set_output_rule(self, rule):
        if isinstance(self.output_rule, TablePlasticityRule):
            self.output_rule.set_rule(rule)
        else:
            raise AssertionError('Currently, set_output_rule() can only be called for table-based plasticity rules.')


    def copy_graph(self, net, input_layer=False, graph=False, output_layer=False):
        """Use the connectivity graphs from another FFBrainNet"""
        if input_layer:
            self.input_layer = net.input_layer
        if graph:
            self.hidden_layers = net.hidden_layers
        if output_layer:
            self.output_layer = net.output_layer


    def update_weights(self, probs, label):
        """
        Update the parameter weights according to the current plasticity rules and the latest firing patterns
        NOTE: We only update weights from one sample at a time
        """

        # Get the prediction for this sample
        prediction = torch.argmax(probs[0])

        # Don't update weights if we're only updating on misclassified examples, and our prediction is correct
        if self.update_scheme.update_misclassified_only and prediction == label:
            return

        # Define our plasticity update rule
        def mult(a, b):
            a *= 1 + self.step_sz * b

        def add(a, b):
            a += self.step_sz * b

        update_func = add if self.options.additive_rule else mult

        # Weight updates
        if self.options.use_graph_rule:
            # Update hidden layer weights according to the plasticity rules

            # For each postsynaptic hidden layer...
            for i in range(1, self.l):
                # Determine the plasticity beta values to use for each entry in the weight matrix
                rule = self.hidden_layer_rules[i]
                betas = rule.hidden_layer_betas(i)

                # Use beta=0 (no update) for any synapse that doesn't exist according to the connectivity graph
                connectivity = self.hidden_layers[i]
                betas *= connectivity

                # Update the weight of each synapse for this layer according to the corresponding beta values
                update_func(self.hidden_weights[i], betas)

        if self.options.use_output_rule:
            # Update output weights according to the plasticity rule

            # Determine the plasticity beta values to use for each entry in the output weight matrix
            betas = self.output_rule.output_betas(prediction, label)

            # Use beta=0 (no update) for any synapse that doesn't exist according to the connectivity graph
            connectivity = self.output_layer
            betas *= connectivity

            if not self.update_scheme.update_all_edges:
                # ONLY update the weights for the prediction output node, and the label output node
                mask = torch.zeros_like(betas)
                mask[prediction] = 1
                mask[label] = 1
                betas *= mask

            # Update the weight of each synapse in the output layer according to the corresponding beta values
            update_func(self.output_weights, betas)


    def forward(self, inputs, labels, epochs, batch, continue_=False):
        """
        Performs a forward pass on inputs and updates the weights after each sample according to the plasticity rules (as enabled by Options)
        Inputs is processed <epochs> times.
        Returns the final loss on all samples after all weight updates
        """

        # When training using fixed rules, continue is False on first sample, True on all subsequent samples
        # i.e. weights are reset initially
        if continue_ is False:
            self.reset_weights(additive=self.options.additive_rule, input_rule=self.options.use_input_rule, output_rule=self.options.use_output_rule)
            self.double()

        # For each inner epoch...
        for epoch in range(1, epochs + 1):
            # For each sample...
            for x, ell in zip(inputs, labels):
                # Get the outputs of this one sample
                # This records the activations that occur
                outputs = self.forward_pass(x.unsqueeze(0))

                # Update the weights using the recorded activations
                self.update_weights(outputs, ell)

        # Generate outputs using final weights
        outputs = self.forward_pass(inputs)

        # Calculate the loss of the final outputs
        if self.update_scheme.mse_loss:
            target = torch.zeros_like(outputs)
            target = target.scatter_(1, labels.unsqueeze(1), 1)
            criterion = nn.MSELoss()
            loss = criterion(outputs, target)
        else:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)

        # Return the loss on the output
        return loss

# ----------------------------------------------------------------------------------------------------------------------
