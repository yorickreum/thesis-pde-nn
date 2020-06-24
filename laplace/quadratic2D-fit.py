import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=2, n_hidden=20, n_output=1)

print("Network")
print(net)

print("\nParameters")
params = list(net.parameters())
print(len(params))
print(params[0].size())


def target_func(x, y):
    return (x - 5.8) ** 2 + (y + 3.2) ** 2


optimizer = torch.optim.Adagrad(net.parameters(), lr=2, lr_decay=.005)
# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.L1Loss()

steps = 100
lin_space = np.linspace(-5, 5, steps)
x = lin_space
y = lin_space
# xy = torch.meshgrid([lin_space, lin_space]) @TODO

xyPerms = []
zPerms = []
for r in itertools.product(x, y):
    xyPerms += [[r[0], r[1]]]
    zPerms += [[target_func(r[0], r[1])]]
xyPerms = torch.tensor(xyPerms, dtype=torch.float)
zPerms = torch.tensor(zPerms, dtype=torch.float)

z = np.zeros((steps, steps))
for ix in range(steps):
    for iy in range(steps):
        z[ix][iy] = target_func(x[ix], y[iy])
z = torch.tensor(z, dtype=torch.float)

assert z[4][5] == target_func(x[4], y[5])

xyPerms, zPerms = Variable(xyPerms), Variable(zPerms)

with torch.no_grad():
    cs = plt.contourf(
        lin_space,
        lin_space,
        z.data.numpy(),
        levels=[i * 10 for i in range(11)],
        extend='both')
    plt.show()

losses = []
for i in range(int(1e4)):
    prediction = net(xyPerms) # xy ist nur ein 100 Elemente-Arry, @TODO 100x100
    loss = loss_func(prediction, zPerms)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    with torch.no_grad():
        losses += [loss.data.numpy()]
        print(loss)

with torch.no_grad():
    zPredicted = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            zPredicted[ix][iy] = \
                net(torch.tensor([x[ix], x[iy]], dtype=torch.float)).data.numpy()
    cs = plt.contourf(
        lin_space,
        lin_space,
        zPredicted,
        levels=[i * 10 for i in range(11)],
        extend='both')
    plt.show()

# @TODO wird aktuell schlechter sobald loss bereits gering
# @TODO --> learning rate dann dynamisch kleiner machen?

plt.plot(
    [i for i in range(len(losses))],
    [i for i in losses]
)
plt.show()
