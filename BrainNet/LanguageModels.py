import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

# 2-layer dense neural network model
class DenseNetwork(nn.Module):
    def __init__(self, embeddings):
        # Expects a pretrained word embedding
        super(DenseNetwork, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze = False)
        self.hidden = nn.Linear(100, 10)
        self.hidden_activation = nn.ReLU()
        self.output = nn.Linear(10, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x) # Apply Glove embedding to sentences (lists of word indices)
        x = torch.sum(x, dim=1).float() # Sum tensors in each row/sentence; Also need to convert to float for training
        x = self.hidden(x)
        x = self.hidden_activation(x)
        x = self.output(x)
        x = self.softmax(x)
        return x # Return 4-vector of class probabilities

# 2-layer RNN model
class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings, mode="RNN"): # mode can be GRU, LSTM, or plain RNN
        # Expects a pretrained word embedding
        super(RecurrentNetwork, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze = False)
        self.hidden_size = 70 # Size of hidden states
        self.num_layers = 2
        if mode == "GRU":
            self.rnn = nn.GRU(100, self.hidden_size, self.num_layers, batch_first=True, dropout = 0)
        elif mode == "LSTM":
            self.rnn = nn.LSTM(100, self.hidden_size, self.num_layers, batch_first=True, dropout = 0)
        else: # default plain RNN
            self.rnn = nn.RNN(100, self.hidden_size, self.num_layers, batch_first=True, dropout = 0)
        self.fc = nn.Linear(self.hidden_size, 4) # Fully connected layer
        self.softmax = nn.Softmax(dim=1)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        batch_size = x.size(0)
        # Get original lengths of sentences
        lengths = []
        for sentence in x:
            padded_indices = (sentence == 0).nonzero()
            if len(padded_indices) > 0:
                lengths.extend(padded_indices[0]) # Get length of sentence
            else:
                lengths.extend(torch.tensor([x.size(1)])) # Sentence has no padding
        lengths = torch.stack(lengths) # Stack sentence lengths into one tensor
        x = self.embedding(x).float() # Apply Glove embedding to sentences (lists of word indices)
        hidden = self.init_hidden(batch_size)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False) #Pack padded sentences
        out, hidden = self.rnn(x, hidden)
        hidden = hidden[self.num_layers - 1] # Get last hidden state of RNN
        hidden = self.fc(hidden)
        hidden = self.softmax(hidden)
        return hidden # Return 4-vector of class probabilities

    def init_hidden(self, batch_size):
        # Generate initial hidden state with zeros
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden