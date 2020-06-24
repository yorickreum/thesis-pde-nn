import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import conservation
from conservation.check_net import check_net
from conservation.net import Net

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device")
print(device)

net = Net(n_feature=2, n_hidden=conservation.HIDDEN, n_output=1)  # schnellste Konvergenz bei 12 hidden
if conservation.DTYPE == torch.double:
    net.double()
net = net.to(device=device)

print("Network")
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())

# optimizer = torch.optim.Adagrad(net.parameters(), lr=2, lr_decay=.005)
# Erfahrung: Bei LBFGS die learning rate auf maximal 0.01 für schnelle Konvergenz
# optimizer = torch.optim.LBFGS(net.parameters(), lr=.01)  # adagrad, lr_decay=.005
optimizer = torch.optim.Adam(net.parameters(), lr=conservation.LR)


def loss_func(input: torch.tensor, prediction: torch.tensor):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=conservation.DTYPE, device=device), 1)
    ts = (input[:, 0:1] ** 2) \
         + (torch.exp(-(input[:, 0:1] ** 2))) \
         + (input[:, 1:2]) * prediction
    ts_d = torch.autograd.grad(
        ts,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    ts_dx = ts_d[:, 0:1]
    ts_dy = ts_d[:, 1:2]
    j = input[:, 0:1] * ts_dx + ts_dy - input[:, 0:1] * input[:, 1:2]
    return torch.sum(j ** 2)


lin_space = np.linspace(0, 1, conservation.DIVISIONS)
x = lin_space
y = lin_space

xyPerms = []
for r in itertools.product(x, y):
    xyPerms += [[r[0], r[1]]]
xyPerms = torch.tensor(xyPerms, dtype=conservation.DTYPE, requires_grad=True, device=device)

# learning_rates = []
startEpoches = time.time()
for i in range(int(conservation.EPOCHS)):
    # net.optimizer = optimizer = torch.optim.LBFGS(net.parameters(), lr=.01 * (1 / (i + 1)))  # adagrad, lr_decay=.005
    optimizer.zero_grad()  # clear gradients for next train
    prediction = net(xyPerms)
    loss = loss_func(xyPerms, prediction)
    with torch.no_grad():
        net.losses += [loss]
        if i % 100 == 0:
            print("step " + str(i) + ": ")
            print(loss)
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
endEpoches = time.time()

torch.save(net, "./model.pt")

print("Runtime of training loop: " + str(endEpoches - startEpoches) + " s")

plt.plot(
    [i for i in range(len(net.losses))],
    [i for i in net.losses]
)
plt.savefig('loss.svg')
plt.savefig('loss.png')
plt.show()

check_net(net)
