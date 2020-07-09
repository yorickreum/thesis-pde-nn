import torch

import oscillation

alpha = 1.611e6
theta_s = 0.287
theta_r = 0.075
beta = 3.96


def theta(h: torch.tensor):
    return ((alpha * (theta_s - theta_r)) / (alpha + torch.abs(h) ** beta)) + theta_r


Ks = 0.009444444444
A = 1.175e6
gamma = 4.74


def K(h: torch.tensor):
    return Ks * A / (A + torch.abs(h) ** gamma)
