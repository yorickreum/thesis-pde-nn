import torch

import laplace_v3


def yorick_delta_gauss(x: torch.Tensor):
    w = .001
    s = 1 / w
    delta = torch.exp(- (s * x) ** 2)
    return delta


def yorick_delta_gauss_relu_cut(x: torch.Tensor):
    w = .1
    s = .1
    delta = torch.exp(- (x / s) ** 2)
    delta = delta * (torch.relu(x + w) * torch.relu(-x + w)) / (w ** 2)
    return delta


def yorick_delta_relu(x: torch.Tensor):
    w = 1
    delta = torch.relu(
        -(torch.relu(x / w) + torch.relu(-x / w))
        + 1)
    return delta


def yorick_delta_relu_sq(x: torch.Tensor):
    w = .01
    delta = (-(x / w) ** 2 + 1) * torch.relu(x + w) * torch.relu(-x + w)
    return delta


def yorick_delta_smooth(x: torch.Tensor):
    w = .1
    s = 4
    delta = torch.relu(
        -(torch.relu(x / w) + torch.relu(-x / w))
        + 1)
    delta = delta ** s
    return delta


def yorick_delta_relu_peak(x: torch.Tensor):
    w = (1 / laplace_v3.DIVISIONS)
    delta = (torch.relu(x + w / 2) / (x + w / 2)) - (torch.relu(x - w / 2) / (x - w / 2))
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
