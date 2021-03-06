# ----------------------------------------------------------------------------------------------------------------------
# COMS 6998_008 Fall 2020: Computation and the Brain
# Final Project
# Group Members: Kartik Balasubramaniam (kb3127), Brett Karopczyc (bjk2161), Vincent Lin (vcl2122), Basile Van Hoorick (bv2279)
# Author(s): Brett Karopczyc
# Portions of this code are based on the BrainNet class (https://github.com/BrainNetwork/BrainNet)
# ----------------------------------------------------------------------------------------------------------------------
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------------

class FFBrainNet(nn.Module):
    """
    FFBrainNet implements a generic ReLU feed-forward ANN with any number of hidden layers of any depth.
    The basic structure of the network is: <input layer> [any number of hidden layers] <output layer>
    Inter-layer connectivity is defined randomly, with a specified probability of an edge existing between nodes of different layers
    The forward propagation of values can be 'capped' at each layer, so that only the highest-valued nodes retain their value

    An FFBrainNet is parameterized by the following:
        n = number of input features (i.e. nodes in the input layer)
        m = number of output labels (i.e. nodes in the output layer)
        l = number of hidden layers (between input and output layers)
        w = common width (# of nodes) of all hidden layers OR array of length l indicating the width of each hidden layer
        p = inter-layer connectivity probability OR array of length l indicating connectivity probability between each
            hidden layer and the preceding layer. NOTE: The output layer is *fully-connected* with the last hidden layer.
        cap = max number of nodes firing at the hidden layers OR array of length l indicating the cap per hidden layer

    Additional parameters:
        use_bias = boolean indicating whether additive bias vectors should be used for all non-input layers
        full_gd = boolean indicating whether the network will be configured to allow gradient descent across all layers.
                  This parameter determines whether weight matrices and bias vectors enable PyTorch's autograd capability
        The following parameters are only meaningful when full_gd is False:
        gd_input = boolean indicating whether the input weights should still support autograd when full_gd is False
        gd_output = boolean indicating whether the output weights & bias should still support autograd when full_gd is False
    """
    def __init__(self, n=10, m=2, l=1, w=100, p=0.5, cap=50, use_bias=False, full_gd=False, gd_input=True, gd_output=False, use_softmax=True):
        super().__init__()

        # Store basic params
        self.n = n
        self.m = m
        self.l = l
        self.use_bias = use_bias
        self.use_softmax = use_softmax

        # Convert per-layer parameters to lists when a scalar is specified
        if isinstance(w, int):
            w = [w] * l     # Common width for all layers
        else:
            assert len(w) == l, "length of w must match l"
        self.w = w

        if isinstance(p, float):
            p = [p] * l     # Common connectivity probability for all layers
        else:
            assert len(p) == l, "length of p must match l"
        self.p = p

        if isinstance(cap, int):
            cap = [cap] * l     # Common cap for all layers
        else:
            assert len(cap) == l, "length of cap must match l"
        self.cap = cap

        # Define random connectivity graphs between layers
        self.input_layer = self.random_bipartite_graph(n, w[0], p[0])
        self.hidden_layers = self.hidden_layer_graphs(w, p)
        self.output_layer = self.random_bipartite_graph(w[-1], m, 1.0)      # Output layer is *fully-connected*

        # Define weight matrices and bias vectors (if requested)
        # NOTE: There is some code duplication with reset_weights() here.
        self.full_gd = full_gd
        self.gd_input = gd_input
        self.gd_output = gd_output
        self.hidden_weights = []
        self.hidden_biases = []
        
        if full_gd:
            # ALL weights and biases should be PyTorch Parameters to enable autograd

            # Input Layer
            self.input_weights = nn.Parameter(torch.rand(w[0], n) - 0.5)

            # Hidden Layers
            for i in range(l):
                # First weight matrix is stored separately in input_weights (for consistency with BrainNet)
                self.hidden_weights.append(None if i==0 else nn.Parameter(torch.rand(w[i], w[i-1]) - 0.5))
                self.hidden_biases.append(nn.Parameter(torch.zeros(w[i])) if use_bias else None)

            # Output Layer
            self.output_weights = nn.Parameter(torch.rand(m, w[-1]) - 0.5)
            self.output_bias = nn.Parameter(torch.zeros(m)) if use_bias else None

        else:
            # DON'T enable autograd for everything
            # If we selectively perform GD for the input and/or output layers, those weights and biases must be
            # PyTorch Parameters to enable autograd.

            # Normally, bias vectors would only be included when learnable via full_GD
            if use_bias:
                print("WARNING: An FFBrainNet with full_gd=False and use_bias=True will likely not learn all bias vectors!")

            # Input Layer
            if gd_input:
                # Enable autograd for input weights
                self.input_weights = nn.Parameter(torch.randn(w[0], n))     # WHY a normal distribution here, but not for full_gd???
            else:
                self.input_weights = torch.randn(w[0], n)

            # Hidden Layers
            for i in range(l):
                # First weight matrix is stored separately in input_weights (for consistency with BrainNet)
                self.hidden_weights.append(None if i==0 else torch.randn(w[i], w[i-1]))     # WHY a normal distribution here, but not for full_gd???
                self.hidden_biases.append(torch.zeros(w[i]) if use_bias else None)

            # Output Layer
            if gd_output:
                # Enable autograd for output weights / bias
                # NOTE: Output bias is NOT a Torch Parameter in BrainNet for some reason
                self.output_weights = nn.Parameter(torch.randn(m, w[-1]))       # WHY a normal distribution here, but not for full_gd???
                self.output_bias = nn.Parameter(torch.zeros(m)) if use_bias else None
            else:
                self.output_weights = torch.randn(m, w[-1])
                self.output_bias = torch.zeros(m) if use_bias else None


    def hidden_layer_graphs(self, w, p):
        """
        Generates random graphs for the inter-layer connectivity
        Returns a list of (l-1) 2D arrays over {0,1}, individually sized per layer according to w, and with edge
        probabilities according to p.

        NOTE: The input layer connectivity is generated separately, so 'None' is returned as the first element of the list
        to keep the indexes in sync with other data structures.
        """
        graphs = [None]     # First random graph is stored in input_layer
        for i in range(1, self.l):
            graphs.append(self.random_bipartite_graph(w[i-1], w[i], p[i]))
        return graphs


    def random_bipartite_graph(self, a, b, p):
        """
        Assuming two adjacent layers of widths a and b, return a bXa array over {0,1}
        where 1 indicates the presence of an edge between corresponding nodes of the two layers.
        The existence of any edge is independently random with probability p.
        """
        adj = torch.rand(b, a).double()
        adj[adj <= 1-p] = 0
        adj[adj > 1-p] = 1
        return adj


    def copy_graph(self, net, input_layer=False, graph=False, output_layer=False):
        """Use the connectivity graphs from another FFBrainNet"""
        if input_layer:
            self.input_layer = net.input_layer
        if graph:
            self.hidden_layers = net.hidden_layers
        if output_layer:
            self.output_layer = net.output_layer


    def forward_pass(self, x):
        """
        Return the output of the network from processing input X, which should have shape: <# of samples> X n
        """

        # Prepare to record the hidden layer (capped) activations during a forward pass
        self.hidden_layer_activations = []

        # This is used in BrainNet to update the hidden vector h during RNN propagation.
        # According to the paper, h is updated as: h <- cap(ReLU(W.h + A.x))
        # where W is the weight matrix of the RNN and A is the weight matrix for the input layer.
        # In other words, every update involves the original activations (stimulus?).
        # self.input = x.clone()

        # We'll perform a more traditional feed-forward propagation of the input:
        # For each layer in the network...
        for weights, connectivity, bias, cap in zip(self.hidden_weights, self.hidden_layers, self.hidden_biases, self.cap):
            # First weight matrix & graph are stored separately
            if weights is None: weights = self.input_weights
            if connectivity is None: connectivity = self.input_layer

            # Compute this layer
            x = self.feed_forward(x, weights, connectivity, bias, cap)

        # Compute the output layer
        out = self.get_output(x)

        # Return the output result
        return out


    def feed_forward(self, x, weights, connectivity, bias, cap):
        """
        Perform a single feed-forward layer on input X of shape: <# of samples> X n
        Overall process: cap(ReLU(weights@X + bias))
        """
        res = torch.mm(weights * connectivity, x.T)
        if self.use_bias:
            res = res + bias[:, None]
        res = F.relu(res)
        res = self.get_cap(res.T, cap)
        return res


    def get_output(self, x):
        """Similar to feed_forward, but softmax instead of ReLU, and no capping"""
        res = torch.mm(self.output_weights * self.output_layer, x.T)
        if self.use_bias:
            res = res + self.output_bias[:, None]

        if self.use_softmax:
            return F.softmax(res.T, dim=1)
        else:
            return res.T


    def get_cap(self, x, cap):
        """Cap input x to largest <cap> values per row, returning 0 in all other indices"""

        # Get the top-k values per row, and copy them to an otherwise zero array
        topk, indices = torch.topk(x, min(cap, x.shape[1]), dim=1)
        res = torch.zeros_like(x)
        res = res.scatter(1, indices, topk)

        # Record activations (0/1) for plasticity rules
        activated = (res != 0).double()
        self.hidden_layer_activations.append(activated.squeeze())

        # Return the capped values
        return res


    def forward(self, inputs):
        """Implement torch.nn.Module's forward(), so a base FFBrainNet can be trained"""
        return self.forward_pass(inputs)


    def reset_weights(self, additive=False, input_rule=False, output_rule=False):
        """
        Called via forward() in plasticity rule-based models.
        Resets the weights and biases of the network's plasticity-based layers.
        Weights & biases are reset for each batch during meta-learning of the plasticity rules.
        NOTE: There is some code duplication with the constructor here.
        """
        assert not self.full_gd, "All weights are torch Parameters, it does not make sense to suddenly replace GD with rules."

        # Always reset hidden layer weights
        self.hidden_weights = []
        if self.use_bias:
            self.hidden_biases = []
        for i in range(self.l):
            if i==0:
                layer_weights = None    # First weight matrix is stored in input_weights
            elif additive:
                layer_weights = torch.zeros(self.w[i], self.w[i - 1])
            else:
                # Multiplicative plasticity rules
                layer_weights = torch.ones(self.w[i], self.w[i - 1])

            self.hidden_weights.append(layer_weights)
            if self.use_bias:
                self.hidden_biases.append(torch.zeros(self.w[i]))

        # Input Layer
        if input_rule:
            assert not self.gd_input, "input_weights is a torch Parameter, it does not make sense to suddenly replace GD with rules."
            self.input_weights = torch.zeros(self.w[0], self.n)

        # Output Layer
        if output_rule:
            assert not self.gd_output, "output_weights is a torch Parameter, it does not make sense to suddenly replace GD with rules."
            if additive:
                self.output_weights = torch.zeros(self.m, self.w[-1])
            else:
                # Multiplicative plasticity rule
                self.output_weights = torch.ones(self.m, self.w[-1])

            if self.use_bias:
                self.output_bias = torch.zeros(self.m)
