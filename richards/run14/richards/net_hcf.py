import torch
from torch import nn

import richards


class HCFNet(nn.Module):
    r""" Network to calculate hydraulic conductivity function (HCF) K
        """
    # optimizer = None
    losses = []

    def __init__(self, n_hidden_neurons):
        super(HCFNet, self).__init__()
        self.hidden = torch.nn.Linear(1, n_hidden_neurons)  # hidden layer
        torch.nn.init.xavier_normal_(self.hidden.weight)
        self.predict = torch.nn.Linear(n_hidden_neurons, 1)  # output layer
        torch.nn.init.xavier_normal_(self.predict.weight)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # tanh activation function for hidden layer
        x = torch.exp(self.predict(x))  # exp output
        return x


hcf_net = HCFNet(n_hidden_neurons=richards.HIDDEN)  # schnellste Konvergenz bei 12 hidden

# @TODO maybe move this part to net constructor
if richards.DTYPE == torch.double:
    hcf_net.double()
hcf_net = hcf_net.to(device=richards.DEVICE)
