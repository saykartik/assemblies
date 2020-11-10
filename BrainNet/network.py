
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from LocalNetBase import *
from BrainNet import BrainNetSequence
from GDNetworks import Regression

class LocalNet(LocalNetBase):
    def __init__(self, n, m, num_v, p, cap, rounds, step_sz=0.01, options = Options(), update_scheme = UpdateScheme()):
        super().__init__(n, m, num_v, p, cap, rounds, step_sz, options = options, update_scheme = update_scheme)

        if self.options.gd_graph_rule:
            self.rule = nn.Parameter(self.rule)
        if self.options.use_input_rule:
            self.input_rule = nn.Parameter(self.input_rule)
        if self.options.gd_output_rule:
            self.output_rule = nn.Parameter(self.output_rule)

class LocalSingleRules(LocalNetBase):
    def __init__(self, n, m, num_v, p, cap, rounds, single_rules, output_rule):
        super().__init__(   n,
                            m,
                            num_v,
                            p,
                            cap,
                            rounds,
                            options = Options(
                                gd_graph_rule = False,
                                gd_output_rule = False,
                                use_output_rule = True,
                                gd_input=False,
                                additive_rule=False
                            ))

        self.first_rule = torch.nn.Parameter(torch.randn(4))
        self.single_rules = single_rules
        self.output_rule = output_rule

    def update_weights(self, probs, label):
        for i in range(self.options.num_graphs):
            prob = probs[i][0]
            prediction = torch.argmax(prob)

            if prediction == label: return

            # update outputs
            if self.options.additive_rule:
                self.output_weights[1 - label] += self.step_sz * self.output_rule[2 * self.activated_rounds[-1].long() + 1]
                self.output_weights[label] += self.step_sz * self.output_rule[2 * self.activated_rounds[-1].long()]
            else:
                self.output_weights[1 - label] *= (1 + self.step_sz * self.output_rule[2 * self.activated_rounds[-1].long() + 1])
                self.output_weights[label] *= (1 + self.step_sz * self.output_rule[2 * self.activated_rounds[-1].long()])

            # update graph weights
            for k in range(self.rounds):
                a1 = self.activated_rounds[k].repeat(1,self.num_v).view(-1, self.num_v)
                a2 = self.activated_rounds[k].view(-1, 1).repeat(1, self.num_v).view(self.num_v, self.num_v)
                act = 2 * a1 + a2

                act *= self.graph
                act = act.long()

                if k == 0:
                    if self.options.additive_rule:
                        self.graph_weights += self.step_sz * self.first_rule[act]
                    else:
                        self.graph_weights *= (1 + self.step_sz * self.first_rule[act])
                else:
                    if self.options.additive_rule:
                        self.graph_weights += self.step_sz * self.single_rules[k - 1][act]
                    else:
                        self.graph_weights *= (1 + self.step_sz * self.single_rules[k - 1][act])



class LocalNetSequence(LocalNet):
    def __init__(self, n, m, num_v, p, cap, rounds, options = Options()):
        super().__init__(n, m, num_v, p, cap, rounds, options = options)
        self.vocab_size = n
        self.network[0] = BrainNetSequence(
                                n,
                                m,
                                num_v = num_v,
                                p = p,
                                cap = cap,
                                rounds = rounds,
                                input_rule = options.use_input_rule,
                                gd_output = self.options.gd_output,
                                gd_input = self.options.gd_input)

    def forward(self, inputs, labels, epochs):
        for i in range(self.options.num_graphs):
            self.reset_weights(additive = self.options.additive_rule, input_rule = self.options.use_input_rule, output_rule = self.options.use_output_rule)
            self.double()
        torch.set_printoptions(precision=3)

        sz = len(inputs)
        criterion = nn.CrossEntropyLoss()

        hidden = self.initHidden()

        loss = 0
        for epoch in range(1, epochs + 1):

            for x,ell in zip(inputs, labels):
                #print(x)
                outputs, hidden = self.network[i](x.double(), hidden.double())
                prediction = torch.argmax(outputs)
                self.update_weights([prediction], ell)

            loss += - torch.log(outputs[labels[-1].long()])

        return loss

    def initHidden(self): return self.initHidden()

class HebbianRule(LocalNetBase):
    # step = amount we change weight of edge per update
    def __init__(self, n, m, num_v, p, cap, rounds, step = 1):
        super().__init__(n, m, num_v, p, cap, rounds,
                        options = Options(
                            num_graphs = 1,
                            rule = None,
                            use_input_rule = False,
                            use_bias_rule = False,
                            use_output_rule = False,
                            additive_rule = True,
                        ))

        n = 2 ** (rounds + 1)
        self.rule = torch.zeros(n ** 2)

        for i in range(n):
            for k in range(n):
                bits_i = self.convert_to_bits(i)
                bits_k = self.convert_to_bits(k)

                for j in range(rounds):
                    if bits_i[j] == 1 and bits_k[j + 1] == 1:
                        self.rule[i * n + k] += step

    def convert_to_bits(self, x):
        arr = []
        for i in range(self.rounds + 1):
            arr.append(x % 2)
            x //= 2
        return arr[::-1] # activation of first round = most significant bit = last entry of array.


class LocalNetRuleModel(LocalNetBase):
    def __init__(self, n, m, num_v, p, cap, rounds, step_sz, options = Options()):
        super().__init__(n, m, num_v, p, cap, rounds, step_sz, options = options)

        self.rule_sz = 4 ** (rounds + 1)
        self.rule_bits = 2 * (rounds + 1) # = log (rule_sz)
        self.conv = [None] * self.rule_sz

        self.edges = (self.graph > 0).nonzero()
        self.output_edges = (self.output_layer > 0).nonzero()
        self.input_edges = (self.input_layer > 0).nonzero()

        # We train this model
        self.rule_model = Regression(self.rule_bits, 100, 1)
        self.rule_model.double()

    def update_weights(self, x, label):
        self.generate_rule()
        super().update_weights(x, label)

    # def update_weights(self, x, label):
    #     self.generate_rule()

    #     step_sz = 1e-3
    #     for i in range(self.options.num_graphs):
    #         outputs = self.network[i](x)
    #         prediction = torch.argmax(outputs)

    #         if True: #prediction != label:
    #             for edge in self.edges:
    #                 a = edge[0]
    #                 b = edge[1]
    #                 act_a = int(self.activated[a])
    #                 act_b = int(self.activated[b])
    #                 self.graph_weights[a][b] += step_sz * self.rule[act_a * (2 ** (self.rounds + 1)) + act_b]

    #             for a in range(self.num_v):
    #                 act_a = int(self.activated[a])
    #                 self.graph_bias[a] += step_sz * self.bias_rule[act_a]

    #             if self.options.use_input_rule:
    #                 for edge in self.input_edges:
    #                     graph_node = edge[0]
    #                     input_node = edge[1]
    #                     act = int(self.activated[graph_node])
    #                     self.input_weights[edge[0]][edge[1]] += step_sz * self.input_rule[act]

    def forward(self, X, y, epochs, batch):
        self.rule = torch.zeros(self.rule_sz)
        return super().forward(X, y, epochs, batch)

    def generate_rule(self):
        for activation_seq in range(self.rule_sz):
            seq = self.convert_to_bits(activation_seq)
            self.rule[activation_seq] = self.rule_model(seq)

    def convert_to_bits(self, x):
        if self.conv[x] is None:
            xx = x
            arr = []
            for i in range(self.rule_bits):
                arr.append(x % 2)
                x //= 2
            self.conv[xx] = torch.tensor(arr).double()
            return self.conv[x]
        else:
            return self.conv[x]

class LocalNetOutputMultiRuleModel(LocalNetBase):
    def __init__(self, n, m, num_v, p, cap, rounds, step_sz=1e-5, options = Options()):
        super().__init__(n, m, num_v, p, cap, rounds, step_sz=1e-5, options = options)

        self.rule_sz = 4 ** (rounds + 1)
        # self.rule_bits = 2 * (rounds + 1) # = log (rule_sz)
        self.rule_bits = self.rounds + 2
        self.conv = [None] * self.rule_sz

        self.edges = (self.graph > 0).nonzero()
        self.output_edges = (self.output_layer > 0).nonzero()
        self.input_edges = (self.input_layer > 0).nonzero()

        # We train this model
        self.output_rule_model = Regression(self.rule_bits, 20, 1)
        self.graph_rule_model = Regression(2 * (self.rounds + 1), 20, 1)
        self.graph_rule_model.hidden.weight.data = torch.zeros_like(self.graph_rule_model.hidden.weight.data)
        self.output_rule_model.double()

    def update_weights(self, x, label):
        prediction = torch.argmax(x[0])

        if self.update_scheme.update_misclassified and prediction == label:
            return

        def mult(a, b):
            a *= 1 + self.step_sz * b
        def add(a, b):
            a += self.step_sz * b

        if self.options.additive_rule:
            update_func = add
        else:
            update_func = mult

        if self.options.use_graph_rule:
          a1 = self.activated.repeat(1,self.num_v).view(-1, self.num_v)
          a2 = self.activated.view(-1, 1).repeat(1, self.num_v).view(self.num_v, self.num_v)
          act = (2 ** (self.rounds + 1)) * a1 + a2
          act *= self.graph

          update_func(self.graph_weights, self.graph_rule_model(self.convert_to_bits(act, b = (self.rounds) * 2 + 1)).squeeze())

        update_func(self.output_weights[prediction], self.output_rule[2 * self.activated.long() + 1])
        update_func(self.output_weights[label], self.output_rule[2 * self.activated.long()])

    def convert_to_bits(self, x, b = None):
        if b is None:
          b = self.rounds
        mask = 2**torch.arange(b,-1,-1)
        return x.int().unsqueeze(-1).bitwise_and(mask).ne(0).byte().double()

class LocalNetMetaRule(LocalNetBase):
    def __init__(self, n, m, num_v, p, cap, rounds, step_sz=1e-5, options = Options()):
        super().__init__(n, m, num_v, p, cap, rounds, step_sz=1e-5, options = options)

        # We train this model
        self.graph_rule_model = Regression(2 * (self.rounds + 1), 20, 1)
        # self.graph_rule_model.hidden.weight.data = torch.zeros_like(self.graph_rule_model.hidden.weight.data)

    def update_weights(self, x, label):
        prediction = torch.argmax(x[0])

        if self.update_scheme.update_misclassified and prediction == label:
            return

        def mult(a, b):
            a *= 1 + self.step_sz * b
        def add(a, b):
            a += self.step_sz * b

        if self.options.additive_rule:
            update_func = add
        else:
            update_func = mult

        if self.options.use_graph_rule:
          a1 = self.activated.repeat(1,self.num_v).view(-1, self.num_v)
          a2 = self.activated.view(-1, 1).repeat(1, self.num_v).view(self.num_v, self.num_v)
          act = (2 ** (self.rounds + 1)) * a1 + a2
          act *= self.graph

          update_func(self.graph_weights, self.graph_rule_model(self.convert_to_bits(act, b = (self.rounds) * 2 + 1)).squeeze())

        update_func(self.output_weights[prediction], self.output_rule[2 * self.activated.long() + 1])
        update_func(self.output_weights[label], self.output_rule[2 * self.activated.long()])

    def convert_to_bits(self, x, b = None):
        if b is None:
          b = self.rounds
        mask = 2**torch.arange(b,-1,-1)
        return x.int().unsqueeze(-1).bitwise_and(mask).ne(0).byte().double()
