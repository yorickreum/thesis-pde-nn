import torch

import oscillation


def get_pde_res(h_val: torch.Tensor, input: torch.Tensor, device=oscillation.DEVICE):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=oscillation.DTYPE, device=device), 1)

    predicted_h_d = torch.autograd.grad(
        h_val,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dt = predicted_h_d[:, 0:1]

    predicted_h_dtd = torch.autograd.grad(
        predicted_h_dt,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dtdt = predicted_h_dtd[:, 0:1]

    residual = \
        1 * predicted_h_dtdt + .1 * h_val  # 1 * predicted_h_dt
    return residual


def get_bc():
    return {
        (0, 0): lambda t, x: 12
    }
