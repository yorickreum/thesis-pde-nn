import itertools

import torch

import oscillation
from oscillation.net_h import PressureHeadNet

import numpy as np

from oscillation.pde import get_pde_res


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

        t = np.linspace(0, 720, oscillation.DIVISIONS)
        z = np.linspace(0, 40, oscillation.DIVISIONS)

        # TODO okay? enforce time boundary only weakly, (ct for ct in t if ct > 0)
        self.tz_pairs_top_boundary = \
            torch.tensor([[it, 40] for it in t],
                         dtype=oscillation.DTYPE, device=self.device,
                         requires_grad=False)
        self.h_top_boundary = 0

        self.tz_pairs_bottom_boundary = \
            torch.tensor([[it, 0] for it in t],
                         dtype=oscillation.DTYPE, device=self.device,
                         requires_grad=False)
        self.h_bottom_boundary = 0

        self.tz_pairs_initial_boundary = \
            torch.tensor([[0, iz] for iz in z],
                         dtype=oscillation.DTYPE, device=self.device,
                         requires_grad=False)
        self.h_initial_boundary = torch.tensor(
            [[np.sin(iz * np.pi / 40)] for iz in z],
            dtype=oscillation.DTYPE, device=self.device)

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
        predicted_h_trial = self.h_net(self.tz_pairs)  # ts(input)
        predicted_h_initial_boundary = self.h_net(self.tz_pairs_initial_boundary)  # ts(tz_pairs_initial_boundary)
        predicted_h_bottom_boundary = self.h_net(self.tz_pairs_bottom_boundary)
        predicted_h_top_boundary = self.h_net(self.tz_pairs_top_boundary)

        residual = get_pde_res(predicted_h_trial, input, device=self.device)

        residual_h_initial_boundary = predicted_h_initial_boundary - self.h_initial_boundary
        residual_h_bottom_boundary = predicted_h_bottom_boundary - self.h_bottom_boundary
        residual_h_top_boundary = predicted_h_top_boundary - self.h_top_boundary
        # residual_q_top_boundary = predicted_q_top_boundary - q_top_boundary

        pde_res = torch.sum(residual ** 2) / len(residual)
        boundary_bottom_res = torch.sum(residual_h_bottom_boundary ** 2) / len(residual_h_bottom_boundary)
        boundary_top_res = torch.sum(residual_h_top_boundary ** 2) / len(residual_h_top_boundary)
        # boundary_top_res = torch.sum(residual_q_top_boundary ** 2) / len(residual_q_top_boundary)
        boundary_initial_res = torch.sum(residual_h_initial_boundary ** 2) / len(residual_h_initial_boundary)

        print("pde_res: " + str(pde_res))
        print("boundary_bottom_res: " + str(boundary_bottom_res))
        print("boundary_top_res: " + str(boundary_top_res))
        print("boundary_initial_res: " + str(boundary_initial_res))

        loss = pde_res + (boundary_bottom_res + boundary_top_res + boundary_initial_res)

        self.h_net.pde_loss += [pde_res]
        self.h_net.bc_initial_losses += [boundary_initial_res]
        self.h_net.bc_top_losses += [boundary_top_res]
        self.h_net.bc_bottom_losses += [boundary_bottom_res]
        self.h_net.losses += [loss]

        return loss

    # @TODO train on mini-batches, see https://stackoverflow.com/a/45118712/8666556
