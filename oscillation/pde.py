import torch

import oscillation
from oscillation.func_lib import yorick_delta


def get_pde_res(h_val: torch.Tensor, input: torch.Tensor, device=oscillation.DEVICE):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=oscillation.DTYPE, device=device), 1)

    predicted_h_d = torch.autograd.grad(
        h_val,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dt = predicted_h_d[:, 0:1]
    predicted_h_dz = predicted_h_d[:, 1:2]

    predicted_h_dtd = torch.autograd.grad(
        predicted_h_dt,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dtdt = predicted_h_dtd[:, 0:1]

    predicted_h_dzd = torch.autograd.grad(
        predicted_h_dz,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dzdz = predicted_h_dzd[:, 1:2]

    residual = 1 * predicted_h_dtdt - 1 * predicted_h_dzdz
    return residual


def get_bc():
    return {
        (0, 0): lambda t, x: 12
    }


def ansatz(input: torch.Tensor, net: torch.Tensor):
    t = input[:, 0:1]
    z = input[:, 1:2]
    return \
        yorick_delta(z) * 0 + \
        yorick_delta(z - 1) * 0 + \
        yorick_delta(t) * 4 * z * (1 - z) + \
        (1 - yorick_delta(z)) * (1 - yorick_delta(z - 1)) * (1 - yorick_delta(t)) * net
