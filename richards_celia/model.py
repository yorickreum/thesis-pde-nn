import itertools

import torch

import richards_celia
from richards_celia.func_lib import theta, K, get_res
from richards_celia.net_h import PressureHeadNet

import numpy as np


class Model:

    def __init__(self, n_hidden, n_neurons, device=richards_celia.DEVICE):
        print("Device: ", device)
        self.device = device

        self.h_net = PressureHeadNet(n_hidden_layers=n_hidden,
                                     n_hidden_neurons=n_neurons,
                                     device=self.device)
        if richards_celia.DTYPE == torch.double:
            self.h_net.double()
        self.h_net = self.h_net.to(device=self.device)

        t = np.linspace(0, 720, richards_celia.DIVISIONS)
        z = np.linspace(0, 40, richards_celia.DIVISIONS)

        # TODO okay? enforce time boundary only weakly, (ct for ct in t if ct > 0)
        self.tz_pairs_top_boundary = \
            torch.tensor([[it, 40] for it in t],
                         dtype=richards_celia.DTYPE, device=self.device,
                         requires_grad=True)
        self.h_top_boundary = -20.7
        # q_top_boundary = 13.69 / (60 * 60)
        # self.h_top_boundary = \
        #    torch.tensor([[-20.7 - 40.8 * np.exp(-1e6 * it)] for it in (ct for ct in t if ct >= 0)],
        #                 dtype=richards_celia.DTYPE, device=self.device)

        self.tz_pairs_bottom_boundary = \
            torch.tensor([[it, 0] for it in t],
                         dtype=richards_celia.DTYPE, device=self.device,
                         requires_grad=True)
        self.h_bottom_boundary = -61.5

        self.tz_pairs_initial_boundary = \
            torch.tensor([[0, iz] for iz in z],
                         dtype=richards_celia.DTYPE, device=self.device,
                         requires_grad=True)
        # self.h_initial_boundary = -61.5
        # = \
        #    torch.tensor([[-61.5 + 40.8 * np.exp(-1e3 * (40 - iz))] for iz in z],
        #                 dtype=richards_celia.DTYPE, device=self.device)
        self.h_initial_boundary = torch.tensor(
            [[-61.5 + 40.8 * np.exp(-1e3 * (40 - iz))] for iz in z],
            dtype=richards_celia.DTYPE, device=self.device)

        # @TODO entferne top und bottom aus tzPairs
        tz_pairs = []
        for r in itertools.product(t, z):
            pair = [r[0], r[1]]
            if not (r[0] == 0 or r[1] == 0 or r[1] == 40):
                tz_pairs += [pair]
        self.tz_pairs = torch.tensor(
            tz_pairs,
            dtype=richards_celia.DTYPE,
            requires_grad=True,
            device=self.device)

    def loss_func(self, input: torch.Tensor):
        ones = torch.unsqueeze(torch.ones(len(input), dtype=richards_celia.DTYPE, device=self.device), 1)

        predicted_h_trial = self.h_net(self.tz_pairs)  # ts(input)
        predicted_h_initial_boundary = self.h_net(self.tz_pairs_initial_boundary)  # ts(tz_pairs_initial_boundary)
        predicted_h_bottom_boundary = self.h_net(self.tz_pairs_bottom_boundary)
        predicted_h_top_boundary = self.h_net(self.tz_pairs_top_boundary)

        residual = get_res(predicted_h_trial, input)

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
