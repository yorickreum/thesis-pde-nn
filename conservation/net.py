import torch
from torch import nn


class Net(nn.Module):
    # optimizer = None
    losses = []

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x