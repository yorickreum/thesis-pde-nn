import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpmath import mp

from check_net import check_net
from laplace_net import Net

mp.dps = 32    # set number of digits

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device")
print(device)

dtype = torch.double

pi_tensor = torch.tensor(mp.pi, dtype=dtype)


net = Net(n_feature=2, n_hidden=8, n_output=1) # schnellste Konvergenz bei 12 hidden
net.double()
net = net.to(device=device)

print("Network")
print(net)

print("\nParameters")
params = list(net.parameters())
print(len(params))
print(params[0].size())

# Erfahrung: Bei LBFGS die learning rate auf maximal 0.01 für schnelle Konvergenz
# optimizer = torch.optim.LBFGS(net.parameters(), lr=.01)  # adagrad, lr_decay=.005
optimizer = torch.optim.Adam(net.parameters(), lr=.005)
# optimizer = torch.optim.Adagrad(net.parameters(), lr=2, lr_decay=.005)


# loss_func = torch.nn.MSELoss()
# loss_func = torch.nn.L1Loss()


def trial_solution(x, y, prediction):
    return y * torch.sin(pi_tensor * x) + x * (x - 1) * y * (y - 1) * prediction


def loss_func(input: torch.tensor, prediction: torch.tensor):
    ts = \
        input[:, 1:2] * torch.sin(pi_tensor * input[:, 0:1]) \
        + input[:, 0:1] * (input[:, 0:1] - 1) * input[:, 1:2] * (input[:, 1:2] - 1) \
        * prediction
    ones = torch.unsqueeze(torch.ones(len(input), dtype=dtype, device=device), 1)
    ts_d = torch.autograd.grad(
        ts,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    ts_dx = ts_d[:, 0:1]
    ts_dy = ts_d[:, 1:2]
    ts_dxx = torch.autograd.grad(
        ts_dx,
        input,
        create_graph=True,
        grad_outputs=ones,
    )
    ts_dxx = (ts_dxx[0])[:, 0:1]
    ts_dyy = (torch.autograd.grad(
        ts_dy,
        input,
        create_graph=True,
        grad_outputs=ones,
    ))
    ts_dyy = (ts_dyy[0])[:, 1:2]
    ts_laplace = ts_dxx + ts_dyy
    return torch.sum(ts_laplace**2)


def closure():
    prediction = net(xyPerms)
    loss = loss_func(xyPerms, prediction)
    with torch.no_grad():
        net.losses += [loss]
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
xyPerms = torch.tensor(xyPerms, dtype=dtype, requires_grad=True, device=device)

# learning_rates = []
for i in range(int(1e5)):
    # net.optimizer = optimizer = torch.optim.LBFGS(net.parameters(), lr=.01 * (1 / (i + 1)))  # adagrad, lr_decay=.005
    print("step " + str(i) + ": ")
    optimizer.zero_grad()  # clear gradients for next train
    loss = closure()
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

torch.save(net, "./model.pt")

plt.plot(
    [i for i in range(len(net.losses))],
    [i for i in net.losses]
)
plt.savefig('loss.svg')
plt.savefig('loss.png')
plt.show()

check_net(net)