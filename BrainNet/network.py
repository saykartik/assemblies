import torch
import torch.nn as nn
from LocalNetBase import * 

class LocalNet(LocalNetBase): 
    '''
        n = # of features
        m = # of possible labels
        num_v = # of nodes in graph
        p = probability that an edge exists in the graph
        cap = choose top 'cap' nodes which fire
        rounds = # of times the graph 'fires'           
    '''
    def __init__(self, n, m, num_v, p, cap, rounds, options = Options(), update_scheme = UpdateScheme()):
        super().__init__(n, m, num_v, p, cap, rounds, options = options, update_scheme = update_scheme)

        if self.options.gd_graph_rule:
            self.rnn_rule = nn.Parameter(self.rnn_rule)
        if self.options.use_input_rule:
            self.input_rule = nn.Parameter(self.input_rule)
        if self.options.gd_output_rule: 
            self.output_rule = nn.Parameter(self.output_rule)
