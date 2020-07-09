import torch
from torch import nn


class Net(nn.Module):
    # optimizer = None
    losses = []

    def __init__(self, n_feature, n_neurons, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_neurons)  # neurons layer
        self.hidden2 = torch.nn.Linear(n_neurons, n_neurons)  # neurons layer
        self.predict = torch.nn.Linear(n_neurons, n_output)  # output layer

    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # activation function for neurons layer
        x = torch.tanh(self.hidden2(x))  # activation function for neurons layer
        x = self.predict(x)  # linear output
        return x