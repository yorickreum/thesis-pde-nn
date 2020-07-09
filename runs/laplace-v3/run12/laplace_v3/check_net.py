import math

import matplotlib.pyplot as plt
import numpy as np
import torch

import laplace_v3
from laplace_v3.func_lib import ansatz

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

plt.style.use('dark_background')


def solution(x: torch.tensor, y: torch.tensor):
    return (1 / (torch.exp(laplace_v3.PI_TENSOR) - torch.exp(-laplace_v3.PI_TENSOR))) * torch.sin(
        laplace_v3.PI_TENSOR * x) * (
                   torch.exp(laplace_v3.PI_TENSOR * y) - torch.exp(- laplace_v3.PI_TENSOR * y))


# def trial_solution(x, y, netxy):
#     return x ** 2 + torch.exp(- x ** 2) + y * netxy


def check_net(net):
    steps = 16
    lin_space = np.linspace(0, 1, steps)
    # zPredicted = np.zeros((steps, steps))
    zNet = np.zeros((steps, steps))
    zSolution = np.zeros((steps, steps))
    for ix in range(steps):
        for iy in range(steps):
            x = torch.tensor(lin_space[ix], dtype=laplace_v3.DTYPE)
            y = torch.tensor(lin_space[iy], dtype=laplace_v3.DTYPE)
            input = torch.tensor([[x, y]], dtype=laplace_v3.DTYPE, device=device)
            netxy = net(input).cpu()
            zNet[iy][ix] = ansatz(input=input, net=netxy)
            # zPredicted[iy][ix] = trial_solution(x, y, netxy)
            zSolution[iy][ix] = solution(x, y)

    levels = np.linspace(zSolution.min(), zSolution.max(), 100)
    cs = plt.contourf(
        lin_space,
        lin_space,
        zSolution,
        levels=levels,
        cmap='rainbow',
        extend='both')
    plt.colorbar(cs)
    plt.title('analytical solution')
    plt.savefig('solution.svg')
    plt.savefig('solution.png')
    plt.show()

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

    # levels = np.linspace(zPredicted.min(), zPredicted.max(), 100)
    # cs = plt.contourf(
    #     lin_space,
    #     lin_space,
    #     zPredicted,
    #     levels=levels,
    #     cmap='rainbow',
    #     extend='both')
    # plt.colorbar(cs)
    # plt.title('prediction')
    # plt.savefig('prediction.svg')
    # plt.savefig('prediction.png')
    # plt.show()

    zDifference = np.abs(zNet - zSolution)
    zDifferenceRel = np.divide(np.abs(zNet - zSolution), zSolution)

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

    zDifferenceRelCleaned = list(
        filter(lambda r: not np.isinf(r),
               zDifferenceRel.flatten())
    )
    print("Max rel. difference: " + str(max(zDifferenceRelCleaned)))
    print("Min rel. difference: " + str(min(zDifferenceRelCleaned)))
    print("Avg rel. difference: " + str(np.mean(zDifferenceRelCleaned)))


if __name__ == '__main__':
    net = torch.load(f"./model.pt")
    check_net(net)
