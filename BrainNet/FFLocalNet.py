import torch
import torch.nn as nn

from FFBrainNet import FFBrainNet
from LocalNetBase import Options, UpdateScheme


class FFLocalNetBase(FFBrainNet):
    """
    n = number of input features
    m = numbe of output labels
    l = number of hidden layers
    w = width of hidden layers OR array of length l indicating width of each hidden layer
    p = inter-layer connectivity probability OR array of length l indicating connectivity probability between each hidden layer and the preceding layer
    cap = max number of nodes firing at the hidden layers OR array of length l containing the cap per hidden layer
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

        # Store params
        self.options = options
        self.update_scheme = update_scheme

        # self.single_rules = torch.randn((rounds, 4))      UNUSED???
        # FOR NOW, ASSUME ALL LAYER WIDTHS AND CAPS ARE IDENTICAL
        self.hidden_layer_rule = torch.randn(2 * (cap + 1) * 2)     # Presyn fired? * Count of incoming fired * Postsyn fired?
        # self.input_rule = torch.randn(2 ** (rounds + 1))          # Currently unused
        self.output_rule = torch.zeros(2 * (cap + 1) * 2)           # Presyn fired? * Count of incoming fired * Is Label node?
        self.step_sz = 0.01

    def get_hidden_layer_rule(self):
        return self.hidden_layer_rule.clone().detach().view(2, 2, self.cap[0]+1)

    def get_output_rule(self):
        return self.output_rule.clone().detach().view(2, 2, self.cap[0]+1)

    def set_hidden_layer_rule(self, rule):
        self.hidden_layer_rule = torch.tensor(rule).flatten().double()

    def set_output_rule(self, rule):
        self.output_rule = rule.clone().detach().flatten().double()

    def copy_graph(self, net, input_layer=False, graph=False, output_layer=False):
        if input_layer:
            self.input_layer = net.input_layer
        if graph:
            self.graphs = net.graphs
        if output_layer:
            self.output_layer = net.output_layer


    def update_weights(self, probs, label):
        prob = probs[0]     # We only update weights on one sample at a time
        prediction = torch.argmax(prob)

        # Skip this if we're only updating on misclassified examples
        if self.update_scheme.update_misclassified and prediction == label:
            return

        # Define our plasticty update rule
        def mult(a, b):
            a *= 1 + self.step_sz * b

        def add(a, b):
            a += self.step_sz * b

        if self.options.additive_rule:
            update_func = add
        else:
            update_func = mult

        # Update input weights
        # CURRENTLY UNUSED - WE DON'T HAVE A DEFINITION FOR AN INPUT PLASTICITY RULE
        if self.options.use_input_rule:
            input_act = self.activated.repeat(1, self.n).view(-1, self.n)  # BUG??? Are the view dims reversed?
            input_act *= self.input_layer
            input_act = input_act.long()

            update_func(self.input_weights, self.input_rule[input_act])

        # Update hidden layer weights according to the plasticity rules
        if self.options.use_graph_rule:
            # For each postsynaptic hidden layer...
            for i in range(1, self.l):
                presyn_width = self.w[i-1]
                postsyn_width = self.w[i]
                presyn_acts = self.hidden_layer_activations[i-1]
                postsyn_acts = self.hidden_layer_activations[i]
                weights = self.graph_weights[i]
                connectivity = self.graphs[i]
                presyn_cap = self.cap[i-1]

                # For each synapse between presynaptic and postsynaptic layers,
                # determine the correct index into the plasticity rule.
                # First, determine the major offset: a function of whether each of the presynatic and postsynaptic neurons fired
                presyn_act_matrix = presyn_acts.repeat(postsyn_width, 1)     # Repeat as rows for each postsynaptic neuron
                postsyn_act_matrix = postsyn_acts.view(-1,1).repeat(1, presyn_width)  # Repeat as cols for each presynaptic neuron
                offset_index = 2 * presyn_act_matrix + postsyn_act_matrix  # Offset index 0-3 for combinations of firings
                offset_matrix = offset_index * (presyn_cap + 1)   # Multiply by number of incoming neurons that could have fired [0 - presynaptic cap]

                # Next, count how many incoming neurons fired per postsynaptic neuron
                incoming_firings = presyn_acts * connectivity   # Broadcasts across rows of connectivty
                incoming_firings = incoming_firings.sum(dim=1, keepdim=True)
                incoming_firings_matrix = incoming_firings.repeat(1, presyn_width)

                # Index into the plasticity rule is the offset + incoming firing count
                rule_idx = offset_matrix + incoming_firings_matrix

                # Ignore any synapse that doesn't exist according to the connectivity graph
                rule_idx *= connectivity
                rule_idx = rule_idx.long()

                # Update the weights of each synapse for this layer
                betas = self.hidden_layer_rule[rule_idx]
                update_func(weights, betas)


        # Update output weights according to the plasticity rules
        if self.options.use_output_rule:
            presyn_acts = self.hidden_layer_activations[-1]
            presyn_width = self.w[-1]
            presyn_cap = self.cap[-1]

            # Count how many incoming neurons fired per label
            incoming_firings = presyn_acts * self.output_layer  # Broadcasts across rows of output_layer
            incoming_firings = incoming_firings.sum(dim=1)
            incoming_firings_matrix = incoming_firings.repeat(1, presyn_width)

            if self.update_scheme.update_all_edges:
                rule_idx = 2 * presyn_acts * (presyn_cap + 1) + incoming_firings[label]
                update_func(self.output_weights[label], self.output_rule[rule_idx])

                for j in range(len(prob)):
                    if j != label:
                        rule_idx = (2 * presyn_acts + 1) * (presyn_cap + 1) + incoming_firings[j]
                        update_func(self.output_weights[j], self.output_rule[rule_idx])
            else:
                # Prediction
                rule_idx = (2 * presyn_acts + 1) * (presyn_cap + 1) + incoming_firings[prediction]
                update_func(self.output_weights[prediction], self.output_rule[rule_idx])

                # Label
                rule_idx = 2 * presyn_acts * (presyn_cap + 1) + incoming_firings[label]
                update_func(self.output_weights[label], self.output_rule[rule_idx])


    def forward(self, inputs, labels, epochs, batch, continue_ = False):
        # Performs a forward pass on inputs AND UPDATES THE WEIGHTS AFTER EACH SAMPLE (as enabled by Options)
        # Returns the final loss on all samples

        # When training using fixed rules, continue is False on first sample, True on all subsequent samples
        # i.e. weights are reset initially
        if continue_ == False:
            self.reset_weights(additive = self.options.additive_rule, input_rule = self.options.use_input_rule, output_rule = self.options.use_output_rule)
            self.double()

        # NOT USED
        # self.output_updates = torch.zeros(self.m, self.num_v)

        # For each epoch...
        for epoch in range(1, epochs + 1):
            # For each sample individually...
            for x,ell in zip(inputs,labels):
                # Get the outputs of this one sample
                # Should record the activations that occur
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

        return loss

class FFLocalNet(FFLocalNetBase):
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        super().__init__(n, m, l, w, p, cap, options=options, update_scheme=update_scheme)

        # Transform rules to Torch 'Parameters' so they're treated as true model params for the Torch Module
        # and updated via an optimizer.step()
        if self.options.gd_graph_rule:
            self.hidden_layer_rule = nn.Parameter(self.hidden_layer_rule)
        # if self.options.use_input_rule:
        #     self.input_rule = nn.Parameter(self.input_rule)
        if self.options.gd_output_rule:
            self.output_rule = nn.Parameter(self.output_rule)
