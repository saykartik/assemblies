# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author: Brett Karopczyc
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
import torch.nn as nn

from FFBrainNet import FFBrainNet
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
            # Update hidden layer weights according to the plasticity rules

            # For each postsynaptic hidden layer...
            for i in range(1, self.l):
                # Determine the plasticity beta values to use for each entry in the weight matrix
                betas = self.hidden_layer_betas(i)

                # Use beta=0 (no update) for any synapse that doesn't exist according to the connectivity graph
                connectivity = self.hidden_layers[i]
                betas *= connectivity

                # Update the weight of each synapse for this layer according to the corresponding beta values
                update_func(self.hidden_weights[i], betas)

        if self.options.use_output_rule:
            # Update output weights according to the plasticity rule

            # Determine the plasticity beta values to use for each entry in the output weight matrix
            betas = self.output_betas(prediction, label)

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

        # For each epoch...
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
# ----------------------------------  Methods to be implemented by subclasses  -----------------------------------------

    def hidden_layer_betas(self, i):
        """
        Returns a 2D array of plasticity beta values for updating the weight matrix between hidden layers i and i-1
        This array should have shape w[i] x w[i-1]
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
