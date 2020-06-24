import numpy as np
import torch

import matplotlib.pyplot as plt
from mpmath import mp

from laplace_net import Net

mp.dps = 32    # set number of digits

dtype = torch.double
pi_tensor = torch.tensor(mp.pi, dtype=dtype)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def solution(x: torch.tensor, y: torch.tensor):
    return (1 / (torch.exp(pi_tensor) - torch.exp(-pi_tensor))) * torch.sin(pi_tensor * x) * (
            torch.exp(pi_tensor * y) - torch.exp(- pi_tensor * y))


def trial_solution(x, y, prediction):
    return y * torch.sin(pi_tensor * x) + x * (x - 1) * y * (y - 1) * prediction


def check_net(net):
    steps = 16
    lin_space = np.linspace(0, 1, steps)
    zNet = np.zeros((steps, steps))
    zPredicted = np.zeros((steps, steps))
    zSolution = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            x = torch.tensor(lin_space[ix], dtype=dtype)
            y = torch.tensor(lin_space[iy], dtype=dtype)
            netxy = net(torch.tensor([x, y], dtype=dtype, device=device)).cpu()
            zNet[iy][ix] = netxy
            zPredicted[iy][ix] = trial_solution(x, y, netxy)
            zSolution[iy][ix] = solution(x, y)

    zDifference = np.abs(zSolution - zPredicted)

    levels = np.linspace(zNet.min(), zNet.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zNet,
        levels=levels,
        cmap='rainbow',
        extend='both')
    plt.colorbar(cs)
    plt.title('network')
    plt.savefig('network.svg')
    plt.savefig('network.png')
    plt.show()

    levels = np.linspace(zPredicted.min(), zPredicted.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zPredicted,
        levels=levels,
        cmap='rainbow',
        extend='both')
    plt.colorbar(cs)
    plt.title('prediction')
    plt.savefig('prediction.svg')
    plt.savefig('prediction.png')
    plt.show()

    levels = np.linspace(zDifference.min(), zDifference.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zDifference,
        levels=levels,
        cmap='rainbow',
        extend='both')
    plt.colorbar(cs)
    plt.title('difference')
    plt.savefig('difference.svg')
    plt.savefig('difference.png')
    plt.show()

    print("Max difference: " + str(zDifference.max()))
    print("Min difference: " + str(zDifference.min()))
    print("Avg difference: " + str(np.mean(zDifference)))


def loadAndCheck():
    # for i in range(20, 22):
    run = "run" + str(23)
    print("Run: " + run)
    net: Net = torch.load("./" + run + "/" + "model.pt", map_location=torch.device('cpu'))
    net.eval()
    check_net(net)