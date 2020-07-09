import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch

import oscillation
from oscillation.func_lib import get_res

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def net_min_max(net, steps, t_min, t_max, z_min, z_max):
    t_lin_space = np.linspace(t_min, t_max, steps)
    z_lin_space = np.linspace(z_min, z_max, steps)

    net_vals = np.zeros((len(z_lin_space), len(t_lin_space)))
    for it in range(len(t_lin_space)):
        for iz in range(len(z_lin_space)):
            t = torch.tensor(t_lin_space[it], dtype=oscillation.DTYPE)
            z = torch.tensor(z_lin_space[iz], dtype=oscillation.DTYPE)
            net_tz = net(torch.tensor([t, z], dtype=oscillation.DTYPE, device=oscillation.DEVICE)).cpu()
            net_vals[iz][it] = net_tz

    return net_vals.min(), net_vals.max()


def print_net(net, steps, t_min, t_max, z_min, z_max):
    t_lin_space = np.linspace(t_min, t_max, steps)
    z_lin_space = np.linspace(z_min, z_max, steps)

    net_vals = np.zeros((len(z_lin_space), len(t_lin_space)))
    for it in range(len(t_lin_space)):
        for iz in range(len(z_lin_space)):
            t = torch.tensor(t_lin_space[it], dtype=oscillation.DTYPE)
            z = torch.tensor(z_lin_space[iz], dtype=oscillation.DTYPE)
            net_tz = net(torch.tensor([t, z], dtype=oscillation.DTYPE, device=oscillation.DEVICE)).cpu()
            net_vals[iz][it] = net_tz

    levels = np.linspace(net_vals.min(), net_vals.max(), 100)
    cs = plt.contourf(
        [x[0] for x in t_lin_space],
        [x[0] for x in z_lin_space],
        net_vals,
        levels=levels,
        cmap='rainbow',
        extend='both')
    plt.colorbar(cs)
    plt.title('network')
    plt.xlabel("Time / days")
    plt.ylabel("depth / cm")
    # plt.savefig('network.svg')
    # plt.savefig('network.png')
    plt.show()


def plot_h(net_h, t, z_bottom, z_top, steps):
    t = torch.tensor(t, dtype=oscillation.DTYPE, device=oscillation.DEVICE)
    z_lin_space = np.linspace(z_bottom, z_top, steps)
    net_vals = np.zeros(len(z_lin_space))

    for iz in range(len(z_lin_space)):
        z = torch.tensor(z_lin_space[iz], dtype=oscillation.DTYPE)
        i_net_h = net_h(torch.tensor([t, z], dtype=oscillation.DTYPE, device=oscillation.DEVICE))
        net_vals[iz] = i_net_h

    plt.plot(
        z_lin_space,
        net_vals
    )
    plt.title(f'h network, t = {t} s')
    plt.xlabel("z / cm")
    plt.ylabel("h / cm")
    # plt.savefig('h.svg')
    # plt.savefig('h.png')
    plt.show()


def plot_h(net_h, t, z_bottom, z_top, steps):
    t = torch.tensor(t, dtype=oscillation.DTYPE, device=oscillation.DEVICE)
    z_lin_space = np.linspace(z_bottom, z_top, steps)
    net_vals = np.zeros(len(z_lin_space))

    for iz in range(len(z_lin_space)):
        z = torch.tensor(z_lin_space[iz], dtype=oscillation.DTYPE)
        i_net_h = net_h(torch.tensor([t, z], dtype=oscillation.DTYPE, device=oscillation.DEVICE))
        net_vals[iz] = i_net_h

    plt.plot(
        z_lin_space,
        net_vals
    )
    plt.title(f'h, t = {t} s')
    plt.xlabel("z / cm")
    plt.ylabel("h / cm")
    # plt.savefig('h.svg')
    # plt.savefig('h.png')
    plt.show()


def plot_res(net, t0, tmax, z0, zmax, steps):
    lin_space_t = np.linspace(t0, tmax, steps)
    lin_space_z = np.linspace(z0, zmax, steps)
    net_vals = np.zeros((len(lin_space_t), len(lin_space_z)))
    net_res = np.zeros((len(lin_space_t), len(lin_space_z)))
    for it in range(len(lin_space_t)):
        for iz in range(len(lin_space_z)):
            t = torch.tensor(lin_space_t[it], dtype=oscillation.DTYPE)
            z = torch.tensor(lin_space_z[iz], dtype=oscillation.DTYPE)
            input = torch.tensor([[t, z]], requires_grad=True, dtype=oscillation.DTYPE, device=oscillation.DEVICE)
            net_tz = net(input).cpu()
            net_vals[iz][it] = net_tz
            net_res[it][iz] = get_res(net_tz, input)

    levels = np.linspace(net_vals.min(), net_vals.max(), 100)
    cs = plt.contourf(
        lin_space_t,
        lin_space_z,
        net_vals,
        levels=levels,
        cmap='rainbow',
        extend='both')
    plt.colorbar(cs)
    plt.title('network')
    # plt.savefig('network.svg')
    # plt.savefig('network.png')
    plt.show()

    levels = np.linspace(net_res.min(), net_res.max(), 100)
    cs = plt.contourf(
        lin_space_t,
        lin_space_z,
        net_res,
        levels=levels,
        cmap='rainbow',
        extend='both')
    plt.colorbar(cs)
    plt.title('residual')
    # plt.savefig('network.svg')
    # plt.savefig('network.png')
    plt.show()


def load_model():
    h_net = torch.load("./h_model.pt", map_location=torch.device('cpu'))
    # plot_h(h_net, 0, 0, 40, 100)
    # plot_h(h_net, 180, 0, 40, 100)
    # plot_h(h_net, 360, 0, 40, 20)
    # plot_h(h_net, 720, 0, 40, 100)
    plot_res(h_net, 0, 720, 0, 40, 20)


if __name__ == "__main__":
    load_model()
