import math

import numpy as np
import torch

import matplotlib.pyplot as plt
from torch import nn


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


def trial_solution(x, y, prediction):
    return y * torch.sin(math.pi * x) + x * (x - 1) * y * (y - 1) * prediction


net = torch.load("./model.pt")
net.eval()

steps = 16
lin_space = np.linspace(0, 1, steps)
x = lin_space
y = lin_space

with torch.no_grad():
    zPredicted = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            x = torch.tensor(lin_space[ix], dtype=torch.float)
            y = torch.tensor(lin_space[iy], dtype=torch.float)
            netxy = net(torch.tensor([x, y], dtype=torch.float)).data.numpy()
            zPredicted[ix][iy] = trial_solution(x, y, netxy)
    levels = np.linspace(zPredicted.min(), zPredicted.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zPredicted,
        levels=levels,
        extend='both')
    plt.show()

plt.plot(
    [i for i in range(len(net.losses))],
    [i for i in net.losses]
)
plt.show()

print("Min loss: " + str(min(net.losses)))
print("Loss in last step: " + str(net.losses[-1]))
