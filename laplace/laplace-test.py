import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import tensor

from differentiate import laplace


def trial_solution(x, y, prediction):
    return y * torch.sin(math.pi * x) + x * (x - 1) * y * (y - 1) * prediction


def solution(x: tensor, y: tensor):
    return (1 / (math.exp(math.pi) - math.exp(-math.pi))) * torch.sin(math.pi * x) * (
                torch.exp(math.pi * y) - torch.exp(- math.pi * y))


xy = tensor([.5, .5], dtype=torch.float, requires_grad=True)
prediction = solution(xy[0], xy[1])
# ts = trial_solution(xy[0], xy[1], prediction)
lap = laplace(prediction, xy)
print(lap)

steps = 16
lin_space = np.linspace(0, 1, steps)
with torch.no_grad():
    zPredicted = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            x = torch.tensor(lin_space[ix], dtype=torch.float)
            y = torch.tensor(lin_space[iy], dtype=torch.float)
            zPredicted[ix][iy] = solution(x, y)

    levels = np.linspace(zPredicted.min(), zPredicted.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zPredicted,
        levels=levels,
        extend='both')
    plt.show()