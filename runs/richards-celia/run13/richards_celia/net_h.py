import torch
from torch import nn


class PressureHeadNet(nn.Module):
    r""" Network to calculate pressure head h
     and hereby solve RRE """
    # optimizer = None
    losses = []

    # net input 1 = t
    # net input 2 = z
    def __init__(self, n_hidden_neurons):
        super(PressureHeadNet, self).__init__()
        self.hidden1 = torch.nn.Linear(2, n_hidden_neurons)  # hidden layer
        torch.nn.init.xavier_normal_(self.hidden1.weight)
        self.hidden2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)  # hidden layer
        torch.nn.init.xavier_normal_(self.hidden2.weight)
        self.predict = torch.nn.Linear(n_hidden_neurons, 1)  # output layer
        torch.nn.init.xavier_normal_(self.predict.weight)

    def forward(self, x):
        x = -torch.tanh(self.hidden1(x))  # tanh activation function for first hidden layer
        # x = torch.tanh(self.hidden2(x))  # linear activation function for second layer @TODO why? okay like this?
        # x = -torch.exp(self.predict(x))  # neg. exp output
        x = self.predict(x)
        return x
