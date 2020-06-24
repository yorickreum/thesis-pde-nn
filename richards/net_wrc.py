import torch
from torch import nn

import richards_celia


class WRCNet(nn.Module):
    r""" Network to calculate water retention curve (WRC) \Theta
        """
    # optimizer = None
    losses = []

    def __init__(self, n_hidden_neurons):
        super(WRCNet, self).__init__()
        self.hidden = torch.nn.Linear(1, n_hidden_neurons)  # hidden layer
        torch.nn.init.xavier_normal_(self.hidden.weight)
        self.predict = torch.nn.Linear(n_hidden_neurons, 1)  # output layer
        torch.nn.init.xavier_normal_(self.predict.weight)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # tanh activation function for hidden layer
        x = torch.sigmoid(self.predict(x))  # sigmoid output
        return x
