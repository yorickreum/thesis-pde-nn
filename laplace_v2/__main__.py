import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import laplace_v2
from laplace_v2.check_net import check_net
from laplace_v2.net import Net

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device")
print(device)

net = Net(n_feature=2, n_hidden=laplace_v2.HIDDEN, n_output=1)  # schnellste Konvergenz bei 12 hidden
if laplace_v2.DTYPE == torch.double:
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
optimizer = torch.optim.Adam(net.parameters(), lr=laplace_v2.LR)

lin_space = np.linspace(0, 1, laplace_v2.DIVISIONS)
x = lin_space
y = lin_space

product = itertools.product(x, y)

xyPerms = [[r[0], r[1]] for r in product]
boundary_1 = filter(lambda p: p[0] == 0 or p[0] == 1 or p[1] == 0, xyPerms)
bc_1_input = [[r[0], r[1]] for r in boundary_1]
boundary_2 = filter(lambda p: p[0] != 0 and p[0] != 1 and p[1] == 1, xyPerms)
bc_2_input = [[r[0], r[1]] for r in boundary_2]
xyPerms = [r for r in xyPerms if (r not in bc_1_input) and (r not in bc_2_input)]

xyPerms = torch.tensor(xyPerms, dtype=laplace_v2.DTYPE, requires_grad=True, device=device)
bc_1_input = torch.tensor(bc_1_input, dtype=laplace_v2.DTYPE, requires_grad=True, device=device)
bc_2_input = torch.tensor(bc_2_input, dtype=laplace_v2.DTYPE, requires_grad=True, device=device)


def loss_func(input: torch.Tensor, bc_1_input: torch.Tensor, bc_2_input: torch.Tensor):
    ts = net(input)
    bc_1 = net(bc_1_input)
    bc_2 = net(bc_2_input)

    ones = torch.unsqueeze(torch.ones(len(input), dtype=laplace_v2.DTYPE, device=device), 1)
    ts_d = torch.autograd.grad(ts, input, create_graph=True, grad_outputs=ones)[0]

    ts_dx = ts_d[:, 0:1]
    ts_dy = ts_d[:, 1:2]

    ts_dxx = torch.autograd.grad(ts_dx, input, create_graph=True, grad_outputs=ones, )
    ts_dxx = (ts_dxx[0])[:, 0:1]

    ts_dyy = (torch.autograd.grad(ts_dy, input, create_graph=True, grad_outputs=ones, ))
    ts_dyy = (ts_dyy[0])[:, 1:2]

    ts_laplace_error = ts_dxx + ts_dyy
    bc_1_error = bc_1
    bc_2_error = bc_2 - torch.sin(laplace_v2.PI_TENSOR * bc_2_input[:, 0:1])

    return torch.sum(ts_laplace_error ** 2) / len(ts_laplace_error) + \
           torch.sum(bc_1_error ** 2) / len(bc_1_error) + \
           torch.sum(bc_2_error ** 2) / len(bc_2_error)


startEpoches = time.time()
for i in range(int(laplace_v2.EPOCHS)):
    # net.optimizer = optimizer = torch.optim.LBFGS(net.parameters(), lr=.01 * (1 / (i + 1)))  # adagrad, lr_decay=.005
    optimizer.zero_grad()  # clear gradients for next train
    loss = loss_func(xyPerms, bc_1_input, bc_2_input)
    with torch.no_grad():
        net.losses += [loss]
        if i % 100 == 0:
            print("step " + str(i) + ": ")
            print(loss)
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    # check if has converged, break early
    with torch.no_grad():
        if i % 10 == 0:
            if len(net.losses) >= 11:
                last_loss_changes = [
                    torch.abs(a_i - b_i)
                    for a_i, b_i in zip(net.losses[-10:], net.losses[-11:-1])
                ]
                if all(llc <= torch.finfo(laplace_v2.DTYPE).eps for llc in last_loss_changes):
                    # or: use max instead of all
                    break
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
