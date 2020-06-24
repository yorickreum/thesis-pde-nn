import math

import numpy as np
import torch

import matplotlib.pyplot as plt
from torch import nn

dtype = torch.double


class Net(nn.Module):
    losses = []

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


def solution(x: torch.tensor, y: torch.tensor):
    return (1 / (math.exp(math.pi) - math.exp(-math.pi))) * torch.sin(math.pi * x) * (
            torch.exp(math.pi * y) - torch.exp(- math.pi * y))


def trial_solution(x, y, prediction):
    return y * torch.sin(math.pi * x) + x * (x - 1) * y * (y - 1) * prediction


net = torch.load("./model.pt", map_location=torch.device('cpu'))
net.eval()

steps = 64
lin_space = np.linspace(0, 1, steps)
x = lin_space
y = lin_space

with torch.no_grad():
    zPredicted = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            x = torch.tensor(lin_space[ix], dtype=dtype)
            y = torch.tensor(lin_space[iy], dtype=dtype)
            netxy = net(torch.tensor([x, y], dtype=dtype)).data.numpy()
            zPredicted[iy][ix] = trial_solution(x, y, netxy)
    levels = np.linspace(zPredicted.min(), zPredicted.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zPredicted,
        levels=levels,
        extend='both')
    plt.colorbar()
    plt.show()

with torch.no_grad():
    zDifference = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            x = torch.tensor(lin_space[ix], dtype=dtype)
            y = torch.tensor(lin_space[iy], dtype=dtype)
            netxy = net(torch.tensor([x, y], dtype=dtype)).data.numpy()
            zDifference[iy][ix] = (solution(x, y) - trial_solution(x, y, netxy))
    levels = np.linspace(zDifference.min(), zDifference.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zDifference,
        levels=levels,
        extend='both')
    plt.colorbar()
    plt.show()

with torch.no_grad():
    zRel = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            x = torch.tensor(lin_space[ix], dtype=dtype)
            y = torch.tensor(lin_space[iy], dtype=dtype)
            netxy = net(torch.tensor([x, y], dtype=dtype)).data.numpy()
            zRel[iy][ix] = (solution(x, y) - trial_solution(x, y, netxy)) / solution(x, y)
    levels = np.linspace(zDifference.min(), zDifference.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zRel,
        levels=levels,
        extend='both')
    plt.colorbar()
    plt.show()

plt.plot(
    [i for i in range(len(net.losses))],
    [i for i in net.losses]
)
plt.show()

print("Min loss: " + str(min(net.losses)))
print("Loss in last step: " + str(net.losses[-1]))
