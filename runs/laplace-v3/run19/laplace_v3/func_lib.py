import torch

import laplace_v3


def yorick_delta_gauss(x: torch.Tensor):
    s = 1e4
    delta = torch.exp(- (s * x) ** 2)
    return delta


def yorick_delta_relu(x: torch.Tensor):
    w = 1
    delta = torch.relu(
        -(torch.relu(x / w) + torch.relu(-x / w))
        + 1)
    return delta


def yorick_delta_relu_sq(x: torch.Tensor):
    w = 1
    delta = (-(x / w) ** 2 + 1) * torch.relu(x + w) * torch.relu(-x + w)
    return delta


def yorick_delta(x: torch.Tensor):
    return yorick_delta_relu(x)


def ansatz(input: torch.Tensor, net: torch.Tensor):
    x = input[:, 0:1]
    y = input[:, 1:2]
    return \
        yorick_delta(x) * 0 + \
        yorick_delta(x - 1) * 0 + \
        yorick_delta(y) * 0 + \
        yorick_delta(y - 1) * torch.sin(laplace_v3.PI_TENSOR * x) + \
        (1 - yorick_delta(x)) * (1 - yorick_delta(x - 1)) * (1 - yorick_delta(y)) * (1 - yorick_delta(y - 1)) * net
