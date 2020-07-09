import torch

import laplace_v3


def yorick_delta(x: torch.Tensor):
    s = 1e2
    delta = torch.exp(- (s * x) ** 2)
    return delta


def ansatz(input: torch.Tensor, net: torch.Tensor):
    x = input[:, 0:1]
    y = input[:, 1:2]
    return \
        yorick_delta(x) * 0 + \
        yorick_delta(x - 1) * 0 + \
        yorick_delta(y) * 0 + \
        yorick_delta(y - 1) * torch.sin(laplace_v3.PI_TENSOR * x) + \
        (1 - yorick_delta(x)) * (1 - yorick_delta(x - 1)) * (1 - yorick_delta(y)) * (1 - yorick_delta(y - 1)) * net