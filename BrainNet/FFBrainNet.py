# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFBrainNet(nn.Module):
    # n = number of input features
    # m = numbe of output labels
    # l = number of hidden layers
    # w = width of hidden layers OR array of length l indicating width of each hidden layer
    # p = inter-layer connectivity probability OR array of length l indicating connectivity probability between each hidden layer and the preceding layer
    # cap = max number of nodes firing at the hidden layers OR array of length l containing the cap per hidden layer
    def __init__(self, n=10, m=2, l=1, w=100, p=0.5, cap=50, full_gd=True, gd_input=True, gd_output=False):
        super().__init__()

        # Store params
        self.n = n
        self.m = m
        self.l = l

        # Convert per-layer params to lists when a scalar is specified
        if isinstance(w, int):
            w = [w] * l
        else:
            assert len(w) == l, "length of w must match l"
        self.w = w

        if isinstance(p, float):
            p = [p] * l
        else:
            assert len(p) == l, "length of p must match l"
        self.p = p

        if isinstance(cap, int):
            cap = [cap] * l
        else:
            assert len(cap) == l, "length of cap must match l"
        self.cap = cap

        # Define the random connectivity graphs
        self.input_layer = self.random_bipartite_graph(n, w[0], p[0])
        self.graphs = self.generate_random_graphs(w, p)
        self.output_layer = self.random_bipartite_graph(w[-1], m, 1.0)  # Assume output layer is fully connected

        # Weights
        if full_gd:
            # All weights and biases should be Torch Parameters to enable autograd

            # Input Layer
            self.input_weights = nn.Parameter(torch.rand(w[0], n) - 0.5)

            # Hidden Layers
            graph_weights = []
            graph_bias = []
            for i in range(l):
                # First weight matrix is stored in input_weights
                graph_weights.append(None if i==0 else nn.Parameter(torch.rand(w[i], w[i-1]) - 0.5))
                graph_bias.append(nn.Parameter(torch.rand(w[i]) - 0.5))
            self.graph_weights = graph_weights
            self.graph_bias = graph_bias

            # Output Layer
            self.output_weights = nn.Parameter(torch.rand(m, w[-1]) - 0.5)
            self.output_bias = nn.Parameter(torch.rand(m) - 0.5)

        else:
            # Don't do gradient descent for everything
            # If we selectively perform GD for the input and/or output layers, those weights and biases
            # must be Torch Parameters to enable autograd.

            # Input Layer
            if gd_input:
                # Still do GD on input layer
                self.input_weights = nn.Parameter(torch.randn(w[0], n))     # WHY a normal distribution here, but not for full_gd???
            else:
                self.input_weights = torch.randn(w[0], n)

            # Hidden Layers
            graph_weights = []
            graph_bias = []
            for i in range(l):
                # First weight matrix is stored in input_weights
                graph_weights.append(None if i==0 else torch.randn(w[i], w[i-1]))     # WHY a normal distribution here, but not for full_gd???
                graph_bias.append(torch.zeros(w[i]))    # WHY not random, like for full_gd???
            self.graph_weights = graph_weights
            self.graph_bias = graph_bias

            # Output Layer
            if gd_output:
                # Still do GD on output layer
                self.output_weights = nn.Parameter(torch.randn(m, w[-1]))       # WHY a normal distribution here, but not for full_gd???
                # NOTE: Output bias is NOT a Torch Parameter in BrainNet for some reason
                self.output_bias = nn.Parameter(torch.zeros(m))       # Why not random, like in full_gd???
            else:
                self.output_weights = torch.randn(m, w[-1])
                self.output_bias = torch.zeros(m)       # Why not random, like in full_gd???


    def generate_random_graphs(self, w, p):
        # Generates random graphs for the inter-hidden-layer connectivity
        # Returns a list of l-1 2D arrays over {0,1}, individually sized per layer
        # The input layer connectivity is generated separately, and so 'None' is returned
        # as the first element of the list (to keep the indexes in sync with other data structures)
        graphs = [None]     # First random graph is stored in input_layer
        for i in range(1, self.l):
            graphs.append(self.random_bipartite_graph(w[i-1], w[i], p[i]))
        return graphs


    def random_bipartite_graph(self, a, b, p):
        # Assuming two adjacent layers of widths a and b, return a bXa 2D array over {0,1}
        # where 1 indicates the presence of an edge between the nodes of the two layers.
        # The existence of any edge is independently random with probability p
        adj = torch.rand(b, a).double()
        adj[adj <= 1-p] = 0
        adj[adj > 1-p] = 1
        return adj


    def forward_pass(self, x):
        # Record the hidden layer (capped) activations during a forward pass
        self.hidden_layer_activations = []

        # self.input = x.clone()    Not sure why this is needed in BrainNet.  Must understand their forward pass...

        for weights, graph, bias, cap in zip(self.graph_weights, self.graphs, self.graph_bias, self.cap):
            if weights is None: weights = self.input_weights
            if graph is None: graph = self.input_layer
            x = self.feed_forward(x, weights, graph, bias, cap)
        out = self.get_output(x)

        return out


    def feed_forward(self, x, weights, graph, bias, cap):
        # Perform a single feed-forward layer on input x
        res = torch.mm(weights * graph, x.T)
        res = res + bias[:, None]
        res = F.relu(res)
        res = self.get_cap(res.T, cap)
        return res


    def get_output(self, x):
        # Similar to feed_forward, but softmax instead of relu, and no capping
        res = torch.mm(self.output_weights * self.output_layer, x.T)
        res = res + self.output_bias[:, None]
        return F.softmax(res.T, dim=1)


    def get_cap(self, x, cap):
        # Cap input x to largest <cap> values, returning 0 in all other indices
        topk, indices = torch.topk(x, min(cap, x.shape[1]), dim=1)
        res = torch.zeros_like(x)
        res = res.scatter(1, indices, topk)

        # Record activations for plasticity rules
        activated = torch.zeros_like(x).scatter(1, indices, 1)
        # self.activated = 2 * self.activated + activated.squeeze()
        self.hidden_layer_activations.append(activated.squeeze())

        return res
