import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from differentiate import laplace


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


net = Net(n_feature=2, n_hidden=8, n_output=1)

print("Network")
print(net)

print("\nParameters")
params = list(net.parameters())
print(len(params))
print(params[0].size())

# Erfahrung: Bei LBFGS die learning rate auf maximal 0.01 für schnelle Konvergenz
optimizer = torch.optim.LBFGS(net.parameters(), lr=.001)   # adagrad, lr_decay=.005
# optimizer = torch.optim.Adagrad(net.parameters(), lr=2, lr_decay=.005)


# loss_func = torch.nn.MSELoss()
# loss_func = torch.nn.L1Loss()


def trial_solution(x, y, prediction):
    return y * torch.sin(math.pi * x) + x * (x - 1) * y * (y - 1) * prediction


def loss_func(input, prediction):
    ret = torch.empty(len(input), 2)
    for index in range(len(input)):
        xy = input[index]
        ts = trial_solution(xy[0], xy[1], prediction[index])
        lap = laplace(ts, xy)
        ret[index] = lap ** 2
    ret = torch.sum(ret)
    return ret


def closure():
    prediction = net(xyPerms)
    loss = loss_func(xyPerms, prediction)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    with torch.no_grad():
        net.losses += [loss.data.numpy()]
        print(loss)
    return loss


steps = 16
lin_space = np.linspace(0, 1, steps)
x = lin_space
y = lin_space
# xy = torch.meshgrid([lin_space, lin_space]) @TODO

xyPerms = []
for r in itertools.product(x, y):
    xyPerms += [[r[0], r[1]]]
xyPerms = torch.tensor(xyPerms, dtype=torch.float, requires_grad=True)

# learning_rates = []
for i in range(int(500)):
    print("step " + str(i) + ": ")
    optimizer.step(closure)  # apply gradients

torch.save(net, "./model.pt")

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
    plt.savefig('prediction.svg')
    plt.savefig('prediction.png')
    plt.show()

# @TODO wird aktuell schlechter sobald loss bereits gering
# @TODO --> learning rate dann dynamisch kleiner machen?

plt.plot(
    [i for i in range(len(net.losses))],
    [i for i in net.losses]
)
plt.savefig('loss.svg')
plt.savefig('loss.png')
plt.show()
