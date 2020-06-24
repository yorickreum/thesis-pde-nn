import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from differentiate import laplace

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device")
print(device)


class Net(nn.Module):
    # optimizer = None
    losses = []

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=2, n_hidden=12, n_output=1)
net.double()
net = net.to(device=device)

print("Network")
print(net)

print("\nParameters")
params = list(net.parameters())
print(len(params))
print(params[0].size())

# Erfahrung: Bei LBFGS die learning rate auf maximal 0.01 für schnelle Konvergenz
optimizer = torch.optim.LBFGS(net.parameters(), lr=.1)  # adagrad, lr_decay=.005


# optimizer = torch.optim.Adagrad(net.parameters(), lr=2, lr_decay=.005)


# loss_func = torch.nn.MSELoss()
# loss_func = torch.nn.L1Loss()


def trial_solution(x, y, prediction):
    return y * torch.sin(math.pi * x) + x * (x - 1) * y * (y - 1) * prediction


def loss_func_old(input, prediction):
    ret = torch.empty(len(input), 2)
    for index in range(len(input)):
        xy = input[index]
        ts = trial_solution(xy[0], xy[1], prediction[index])
        lap = laplace(ts, xy)
        ret[index] = lap ** 2
    ret = torch.sum(ret)
    return ret


def loss_func(input, prediction):
    ts = ts_term1 + ts_term2 * prediction
    ones = torch.unsqueeze(torch.ones(len(input), dtype=torch.double, device=device), 1)
    ts_d = torch.autograd.grad(
        ts,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    ts_dx = ts_d[:, 0:1]
    ts_dy = ts_d[:, 1:2]
    ts_dxx = (torch.autograd.grad(
        ts_dx,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0])[:, 0:1]
    ts_dyy = (torch.autograd.grad(
        ts_dy,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0])[:, 1:2]
    return torch.sum(ts_dxx ** 2) + torch.sum(ts_dyy ** 2)


def closure():
    prediction = net(xyPerms)
    loss = loss_func(xyPerms, prediction)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    with torch.no_grad():
        net.losses += [loss]
        print(loss)
    return loss


steps = 64
lin_space = np.linspace(0, 1, steps)
x = lin_space
y = lin_space
# xy = torch.meshgrid([lin_space, lin_space]) @TODO

xyPerms = []
for r in itertools.product(x, y):
    xyPerms += [[r[0], r[1]]]
xyPerms = torch.tensor(xyPerms, dtype=torch.double, requires_grad=True, device=device)


def ts_1(xy):
    return xy[1] * torch.sin(math.pi * xy[0])


def ts_2(xy):
    return xy[0] * (xy[0] - 1) * xy[1] * (xy[1] - 1)


ts_term1 = []
ts_term2 = []
for i in range(len(xyPerms)):
    ts_term1 += [[ts_1(xyPerms[i])]]
    ts_term2 += [[ts_2(xyPerms[i])]]
ts_term1 = torch.tensor(ts_term1, dtype=torch.double, requires_grad=True, device=device)
ts_term2 = torch.tensor(ts_term2, dtype=torch.double, requires_grad=True, device=device)

# learning_rates = []
for i in range(int(1e5)):
    # net.optimizer = optimizer = torch.optim.LBFGS(net.parameters(), lr=.01 * (1 / (i + 1)))  # adagrad, lr_decay=.005
    print("step " + str(i) + ": ")
    optimizer.step(closure)  # apply gradients

torch.save(net, "./model.pt")

with torch.no_grad():
    zPredicted = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            x = torch.tensor(lin_space[ix], dtype=torch.double)
            y = torch.tensor(lin_space[iy], dtype=torch.double)
            netxy = net(torch.tensor([x, y], dtype=torch.double, device=device)).cpu().data.numpy()
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
