import math

import torch
from torch import nn

import richards
from richards.net_hcf import hcf_net
from richards.net_wrc import wrc_net


class Net(nn.Module):
    r""" Network to calculate matric potential \psi
     and hereby solve RRE """
    # optimizer = None
    losses = []

    def __init__(self, n_hidden_neurons):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(2, n_hidden_neurons)  # hidden layer
        torch.nn.init.xavier_normal_(self.hidden.weight)
        # net input 1 --> t
        # net input 2 --> z
        self.predict = torch.nn.Linear(n_hidden_neurons, 1)  # output layer
        torch.nn.init.xavier_normal_(self.predict.weight)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # tanh activation function for hidden layer
        x = -torch.exp(self.predict(x))  # neg. exp output
        return x


def loss_func(input: torch.tensor,
              predicted_matrix_potential: torch.tensor,
              theta: torch.tensor):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=richards.DTYPE, device=richards.DEVICE), 1)

    # Check for NaN
    assert (input == input).any()
    assert (predicted_matrix_potential == predicted_matrix_potential).any()
    assert (theta == theta).any()

    predicted_matrix_potential_d = torch.autograd.grad(
        predicted_matrix_potential,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    # predicted_matrix_potential_dt = predicted_matrix_potential_d[:, 0:1]
    predicted_matrix_potential_dz = predicted_matrix_potential_d[:, 1:2]
    predicted_matrix_potential_dzz = torch.autograd.grad(
        predicted_matrix_potential_dz,
        input,
        create_graph=True,
        grad_outputs=ones,
    )
    # predicted_matrix_potential_dzt = (predicted_matrix_potential_dzz[0])[:, 0:1]
    predicted_matrix_potential_dzz = (predicted_matrix_potential_dzz[0])[:, 1:2]

    predicted_matrix_potential_log = torch.log(-predicted_matrix_potential)  # minus cause matrix potential is negative
    assert (predicted_matrix_potential_log == predicted_matrix_potential_log).any()

    predicted_hcf = hcf_net(predicted_matrix_potential_log)
    assert (predicted_hcf == predicted_hcf).any()
    predicted_hcf_d = torch.autograd.grad(
        predicted_hcf,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_hcf_dz = predicted_hcf_d[:, 1:2]

    predicted_wrc = wrc_net(predicted_matrix_potential_log)
    assert (predicted_wrc == predicted_wrc).any()
    predicted_wrc_d = torch.autograd.grad(
        predicted_wrc,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_wrc_dt = predicted_wrc_d[:, 0:1]

    residual = \
        predicted_wrc_dt \
        - predicted_hcf_dz * predicted_matrix_potential_dz \
        - predicted_hcf * predicted_matrix_potential_dzz \
        - predicted_matrix_potential_dz

    # fitting_error = torch.unsqueeze(torch.zeros(len(input), dtype=richards.DTYPE, device=richards.DEVICE), 1)
    fitting_error = predicted_wrc - theta
    assert (fitting_error == fitting_error).any()

    loss = torch.sum(fitting_error ** 2) + torch.sum(residual ** 2)
    assert (loss == loss).any()

    return loss


net = Net(n_hidden_neurons=richards.HIDDEN)  # schnellste Konvergenz bei 12 hidden

# @TODO maybe move this part to net constructor
if richards.DTYPE == torch.double:
    net.double()
net = net.to(device=richards.DEVICE)
