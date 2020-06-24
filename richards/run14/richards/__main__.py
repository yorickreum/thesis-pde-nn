import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
import torch

import richards
from richards.check_net import check_net
from richards.net_psi import loss_func, net
from richards.net_hcf import hcf_net
from richards.net_wrc import wrc_net

print("Device")
print(richards.DEVICE)

print("Network")
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())

# optimizer = torch.optim.Adagrad(net.parameters(), lr=2, lr_decay=.005)
# Erfahrung: Bei LBFGS die learning rate auf maximal 0.01 für schnelle Konvergenz
# optimizer = torch.optim.LBFGS(net.parameters(), lr=.01)  # adagrad, lr_decay=.005
parameters = set()
parameters |= set(net.parameters())
parameters |= set(wrc_net.parameters())
parameters |= set(hcf_net.parameters())
optimizer = torch.optim.Adam(parameters, lr=richards.LR)

# t_lin_space = np.linspace(0, 3, 144)
# z_lin_space = np.linspace(-20, 0, 100)
# tzPerms = []
# for r in itertools.product(t_lin_space, z_lin_space):
#     tzPerms += [[r[0], r[1]]]
# tzPerms = torch.tensor(tzPerms, dtype=richards.DTYPE, requires_grad=True, device=richards.DEVICE)

data = pandas.read_csv(f"./loam_nod.csv")
rows = None  # None --> all rows

t = data['time'].values[:rows, None]
z = data['depth'].values[:rows, None]
tzPairs = np.concatenate((t, z), 1)
tzPairs = torch.tensor(tzPairs,
                       dtype=richards.DTYPE,
                       requires_grad=True,
                       device=richards.DEVICE)

theta = data['theta'].values[:rows, None]
theta = torch.tensor(theta,
                     dtype=richards.DTYPE,
                     requires_grad=False,
                     device=richards.DEVICE)

# learning_rates = []
startEpoches = time.time()
for i in range(int(richards.EPOCHS)):
    # net.optimizer = optimizer = torch.optim.LBFGS(net.parameters(), lr=.01 * (1 / (i + 1)))  # adagrad, lr_decay=.005
    optimizer.zero_grad()  # clear gradients for next train
    prediction = net(tzPairs)
    loss = loss_func(tzPairs, prediction, theta)
    with torch.no_grad():
        net.losses += [loss]
        if i % richards.LOGSTEPS == 0:
            print("step " + str(i) + ": ")
            print(loss)
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    # @TODO check for a better way
    # To embrace the monotonicity of WRC and HCF, the weight parameters WTheta and WK are constrained to be
    # non-negative so that fTheta and fK are increasingly monotonic functions with respect to the predicted matric
    # potential. The monotonicity honors the physical nature of WRC and HCF of all soils.
    for p in wrc_net.parameters():
        p.data.clamp_(0)
    for p in hcf_net.parameters():
        p.data.clamp_(0)
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
