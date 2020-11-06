# Imports
import torch
import torch.nn as nn

from FFBrainNet import FFBrainNet
from LocalNetBase import Options, UpdateScheme

# ----------------------------------------------------------------------------------------------------------------------

class FFLocalNetBase(FFBrainNet):
    """
    This class extends FFBrainNet to add support for local plasticity rules, in a manner analogous to LocalNetBase.
    This support includes learning and using plasticity rules for both hidden layers and the output layer.

    NOTE: This class currently implements a SINGLE hidden-layer plasticity rule that applies to all hidden layer weight updates
    The single rule used to update a synapse between nodes i and j can be represented as a table of size 2 * 2 * (cap + 1):
        <node i fired?> * <node j fired?> * <number of incoming nodes to j that fired>
    As a result, we currently assume that all hidden layers have the same cap (which determines the size of the common rule)
    Further development can enhance the number and/or structure of the plasticity rule(s) used.

    An instance of FFLocalNetBase shares the same network structure as FFBrainNet:
        <input layer> [any number of hidden layers] <output layer>, which random inter-layer connectivity

    The basic parameterization of the class is the same as FFBrainNet:
        n = number of input features (i.e. nodes in the input layer)
        m = number of output labels (i.e. nodes in the output layer)
        l = number of hidden layers (between input and output layers)
        w = common width (# of nodes) of all hidden layers OR array of length l indicating the width of each hidden layer
        p = inter-layer connectivity probability OR array of length l indicating connectivity probability between each
            hidden layer and the preceding layer. NOTE: The output layer is *fully-connected* with the last hidden layer.
        cap = max number of nodes firing at the hidden layers OR array of length l indicating the cap per hidden layer

    In addition, the Options and UpdateScheme structures from LocalNetBase were reused to control how the class is trained.
    """
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        super().__init__(n=n,
                         m=m,
                         l=l,
                         w=w,
                         p=p,
                         cap=cap,
                         gd_input=options.gd_input,
                         gd_output=options.gd_output)

        # Store additional params
        self.options = options
        self.update_scheme = update_scheme
        self.step_sz = 0.01

        # Create plasticity rules:

        # Hidden layer rule
        # (assumes all hidden layer caps are identical)
        self.hidden_layer_rule = torch.randn(2 * 2 * (cap + 1))     # Presyn fired? * Postsyn fired? * Count of incoming fired

        # Output layer rule
        self.output_rule = torch.zeros(2 * 2 * (cap + 1))           # Presyn fired? * Postsyn is Label node? * Count of incoming fired


    def get_hidden_layer_rule(self):
        """Return the hidden layer plasticity rule as shape 2 * 2 * (cap+1)"""
        return self.hidden_layer_rule.clone().detach().view(2, 2, self.cap[0]+1)

    def get_output_rule(self):
        """Return the output layer plasticity rule as shape 2 * 2 * (cap+1)"""
        return self.output_rule.clone().detach().view(2, 2, self.cap[0]+1)

    def set_hidden_layer_rule(self, rule):
        self.hidden_layer_rule = torch.tensor(rule).flatten().double()

    def set_output_rule(self, rule):
        self.output_rule = rule.clone().detach().flatten().double()

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
        if self.update_scheme.update_misclassified and prediction == label:
            return

        # Define our plasticity update rule
        def mult(a, b):
            a *= 1 + self.step_sz * b

        def add(a, b):
            a += self.step_sz * b

        update_func = add if self.options.additive_rule else mult

        # Weight updates
        if self.options.use_graph_rule:
            # Update hidden layer weights according to the common plasticity rule

            # For each postsynaptic hidden layer...
            for i in range(1, self.l):
                # Get details of the presynaptic and postsynaptic layers, their connectivity, and their latest firing patterns
                presyn_width = self.w[i-1]
                postsyn_width = self.w[i]
                presyn_acts = self.hidden_layer_activations[i-1]
                postsyn_acts = self.hidden_layer_activations[i]
                weights = self.hidden_weights[i]
                connectivity = self.hidden_layers[i]
                presyn_cap = self.cap[i-1]

                # For each synapse between presynaptic and postsynaptic layers,
                # determine the correct index into the plasticity rule (which is stored as a flat vector).
                # The index is: (2 * presyn fired? + postsyn fired?) * (cap + 1) + number of incoming fired
                # In other words, the rule is structured as 4 consecutive subarrays of length (cap + 1),
                # corresponding to the 4 possible combos of the presyn and postsyn nodes firing.

                # First, determine the major offset: a function of whether each of the presynatic and postsynaptic neurons fired
                presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)     # Repeat as rows for each postsynaptic neuron
                postsyn_act_matrix = postsyn_acts.view(-1,1).repeat(1, presyn_width)  # Repeat as cols for each presynaptic neuron
                offset_index = 2 * presyn_act_matrix + postsyn_act_matrix  # Offset index 0-3 for combinations of firings
                offset_matrix = offset_index * (presyn_cap + 1)   # Multiply by number of incoming neurons that could have fired [0 - presynaptic cap]

                # Next, count how many incoming neurons fired per postsynaptic neuron
                incoming_firings = presyn_acts * connectivity   # Broadcasts across rows of connectivity, yielding activity for each postsynaptic neuron
                incoming_firings = incoming_firings.sum(dim=1, keepdim=True)    # Count the number of incoming nodes that fired
                incoming_firings_matrix = incoming_firings.repeat(1, presyn_width)  # Repeat as cols for each presynaptic neuron

                # Index into the plasticity rule is the offset + incoming firing count
                rule_idx = offset_matrix + incoming_firings_matrix

                # Ignore any synapse that doesn't exist according to the connectivity graph
                rule_idx *= connectivity
                rule_idx = rule_idx.long()

                # Update the weight of each synapse for this layer according to the corresponding beta values
                betas = self.hidden_layer_rule[rule_idx]
                update_func(weights, betas)


        if self.options.use_output_rule:
            # Update output weights according to the plasticity rule

            # Get the latest firing pattern of the presynaptic layer and its cap
            # (for the output rule, this is the last hidden layer)
            presyn_acts = self.hidden_layer_activations[-1]
            presyn_cap = self.cap[-1]

            # Count how many incoming neurons fired per label
            incoming_firings = presyn_acts * self.output_layer  # Broadcasts across rows of output_layer
            incoming_firings = incoming_firings.sum(dim=1)

            # The structure of the output rule is similar to the hidden-layer rule.
            # The index is: (2 * presyn fired? + postsyn is NOT the label?) * (cap + 1) + number of incoming fired

            if self.update_scheme.update_all_edges:
                # Every edge of the output layer should be updated

                # Update all edges for the label output node
                rule_idx = (2 * presyn_acts + 0) * (presyn_cap + 1) + incoming_firings[label]
                update_func(self.output_weights[label], self.output_rule[rule_idx.long()])

                # Update all edges for every other output node
                for j in range(self.m):
                    if j != label:
                        rule_idx = (2 * presyn_acts + 1) * (presyn_cap + 1) + incoming_firings[j]
                        update_func(self.output_weights[j], self.output_rule[rule_idx.long()])
            else:
                # ONLY update the weights for the prediction output node, and the label output node

                # Prediction
                rule_idx = (2 * presyn_acts + 1) * (presyn_cap + 1) + incoming_firings[prediction]
                update_func(self.output_weights[prediction], self.output_rule[rule_idx.long()])

                # Label
                rule_idx = (2 * presyn_acts + 0) * (presyn_cap + 1) + incoming_firings[label]
                update_func(self.output_weights[label], self.output_rule[rule_idx.long()])


    def forward(self, inputs, labels, epochs, batch, continue_ = False):
        """
        Performs a forward pass on inputs and updates the weights after each sample according to the plasticity rules (as enabled by Options)
        Inputs is processed <epochs> times.
        Returns the final loss on all samples after all weight updates
        """

        # When training using fixed rules, continue is False on first sample, True on all subsequent samples
        # i.e. weights are reset initially
        if continue_ == False:
            self.reset_weights(additive = self.options.additive_rule, input_rule = self.options.use_input_rule, output_rule = self.options.use_output_rule)
            self.double()

        # For each epoch...
        for epoch in range(1, epochs + 1):
            # For each sample...
            for x,ell in zip(inputs,labels):
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

class FFLocalNet(FFLocalNetBase):
    """
    FFLocalNet is a subclass of FFLocalNetBase that enables learning of plasticity rules via gradient descent
    When specified, the plasticity rules are treated as PyTorch Parameters, and so have autograd enabled and can be optimized
    """
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        super().__init__(n, m, l, w, p, cap, options=options, update_scheme=update_scheme)

        # Convert plasticity rules to Torch 'Parameters' so they're treated as true model params for the Torch Module
        # and updated via an optimizer.step()
        if self.options.gd_graph_rule:
            self.hidden_layer_rule = nn.Parameter(self.hidden_layer_rule)
        if self.options.gd_output_rule:
            self.output_rule = nn.Parameter(self.output_rule)

# ----------------------------------------------------------------------------------------------------------------------
