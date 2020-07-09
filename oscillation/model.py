import itertools

import torch

import oscillation
from oscillation.net_h import PressureHeadNet

import numpy as np

from oscillation.pde import get_pde_res, ansatz


class Model:

    def __init__(self, n_hidden, n_neurons, device=oscillation.DEVICE):
        print("Device: ", device)
        self.device = device

        self.h_net = PressureHeadNet(n_hidden_layers=n_hidden,
                                     n_hidden_neurons=n_neurons,
                                     device=self.device)
        if oscillation.DTYPE == torch.double:
            self.h_net.double()
        self.h_net = self.h_net.to(device=self.device)

        t = np.linspace(0, 10, oscillation.DIVISIONS)
        z = np.linspace(0, 1.1, oscillation.DIVISIONS)

        # @TODO entferne top und bottom aus tzPairs
        tz_pairs = []
        for r in itertools.product(t, z):
            pair = [r[0], r[1]]
            # if not (r[0] == 0 or r[1] == 0 or r[1] == 40):
            tz_pairs += [pair]
        self.tz_pairs = torch.tensor(
            tz_pairs,
            dtype=oscillation.DTYPE,
            requires_grad=True,
            device=self.device)

    def loss_func(self, input: torch.Tensor):
        predicted_h = self.h_net(self.tz_pairs)  # ts(input)
        predicted_h_trial = ansatz(input, predicted_h)
        residual = get_pde_res(predicted_h_trial, input, device=self.device)
        pde_res = torch.sum(residual ** 2) / len(residual)
        print("pde_res: " + str(pde_res))
        loss = pde_res
        self.h_net.pde_loss += [pde_res]
        return loss

    # @TODO train on mini-batches, see https://stackoverflow.com/a/45118712/8666556
