import matplotlib.pyplot as plt
import numpy as np
import torch

import conservation

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def solution(x: torch.tensor, y: torch.tensor):
    return x*(y-1) + x**2 * torch.exp(-2*y) + torch.exp(-x**2 * torch.exp(-2 * y)) + x * torch.exp(-y)


def trial_solution(x, y, netxy):
    return x ** 2 + torch.exp(- x ** 2) + y * netxy


def check_net(net):
    steps = 16
    lin_space = np.linspace(0, 1, steps)
    zPredicted = np.zeros((steps, steps))
    zNet = np.zeros((steps, steps))
    zSolution = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            x = torch.tensor(lin_space[ix], dtype=conservation.dtype)
            y = torch.tensor(lin_space[iy], dtype=conservation.dtype)
            netxy = net(torch.tensor([x, y], dtype=conservation.dtype, device=device)).cpu()
            zNet[iy][ix] = netxy
            zPredicted[iy][ix] = trial_solution(x, y, netxy)
            zSolution[iy][ix] = solution(x, y)

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

    zDifference = np.abs(zPredicted - zSolution)
    zDifferenceRel = np.divide(np.abs(zPredicted - zSolution), zSolution)

    levels = np.linspace(zDifference.min(), zDifference.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zDifference,
        levels=levels,
        cmap='rainbow',
        extend='both')
    plt.colorbar(cs)
    plt.title('difference to analytical solution')
    plt.savefig('difference.svg')
    plt.savefig('difference.png')
    plt.show()

    print("Max difference: " + str(zDifference.max()))
    print("Min difference: " + str(zDifference.min()))
    print("Avg difference: " + str(np.mean(zDifference)))

    print("Max rel. difference: " + str(zDifferenceRel.max()))
    print("Min rel. difference: " + str(zDifferenceRel.min()))
    print("Avg rel. difference: " + str(np.mean(zDifferenceRel)))