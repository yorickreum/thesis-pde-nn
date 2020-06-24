import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch

import richards_celia

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
            t = torch.tensor(t_lin_space[it], dtype=richards_celia.DTYPE)
            z = torch.tensor(z_lin_space[iz], dtype=richards_celia.DTYPE)
            net_tz = net(torch.tensor([t, z], dtype=richards_celia.DTYPE, device=richards_celia.DEVICE)).cpu()
            net_vals[iz][it] = net_tz

    return net_vals.min(), net_vals.max()


def print_net(net, steps, t_min, t_max, z_min, z_max):
    t_lin_space = np.linspace(t_min, t_max, steps)
    z_lin_space = np.linspace(z_min, z_max, steps)

    net_vals = np.zeros((len(z_lin_space), len(t_lin_space)))
    for it in range(len(t_lin_space)):
        for iz in range(len(z_lin_space)):
            t = torch.tensor(t_lin_space[it], dtype=richards_celia.DTYPE)
            z = torch.tensor(z_lin_space[iz], dtype=richards_celia.DTYPE)
            net_tz = net(torch.tensor([t, z], dtype=richards_celia.DTYPE, device=richards_celia.DEVICE)).cpu()
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
    plt.savefig('network.svg')
    plt.savefig('network.png')
    plt.show()


def print_wrc_psi(net_wrc, steps, psi_min, psi_max):
    psi_lin_space = np.linspace(psi_min, psi_max, steps)
    net_vals = np.zeros(len(psi_lin_space))

    for ipsi in range(len(psi_lin_space)):
        psi = torch.tensor(psi_lin_space[ipsi], dtype=richards_celia.DTYPE)
        net_psi = net_wrc(torch.tensor([psi], dtype=richards_celia.DTYPE, device=richards_celia.DEVICE)).cpu()
        net_vals[ipsi] = net_psi

    plt.plot(
        psi_lin_space,
        net_vals
    )
    plt.title('WRC network')
    plt.xlabel("psi / -")
    plt.ylabel("wrc / ?")
    plt.savefig('wrc.svg')
    plt.savefig('wrc.png')
    plt.show()

def print_hcf_psi(net_hcf, steps, psi_min, psi_max):
    psi_lin_space = np.linspace(psi_min, psi_max, steps)
    net_vals = np.zeros(len(psi_lin_space))

    for ipsi in range(len(psi_lin_space)):
        psi = torch.tensor(psi_lin_space[ipsi], dtype=richards_celia.DTYPE)
        net_psi = net_hcf(torch.tensor([psi], dtype=richards_celia.DTYPE, device=richards_celia.DEVICE)).cpu()
        net_vals[ipsi] = net_psi

    plt.plot(
        psi_lin_space,
        net_vals
    )
    plt.title('HCF network')
    plt.xlabel("hcf / -")
    plt.ylabel("hcf / ?")
    plt.savefig('wrc.svg')
    plt.savefig('wrc.png')
    plt.show()


def check_model(psi_net, wrc_net, hcf_net):
    data = pandas.read_csv(f"./train_data.csv")
    rows = None  # None --> all rows
    t = data['t'].values[:rows, None]
    z = data['z'].values[:rows, None]
    theta = data['theta_train'].values[:rows, None]

    t_min, t_max = min(t), max(t)
    z_min, z_max = min(z), max(z)

    psi_min, psi_max = net_min_max(psi_net, 100, t_min, t_max, z_min, z_max)

    print_net(psi_net, 100, t_min, t_max, z_min, z_max)
    print_wrc_psi(wrc_net, 100, psi_min, psi_max)
    print_hcf_psi(hcf_net, 100, psi_min, psi_max)


def load_model():
    psi_net = torch.load("./psi_model.pt", map_location=torch.device('cpu'))
    wrc_net = torch.load("./wrc_model.pt", map_location=torch.device('cpu'))
    hcf_net = torch.load("./hcf_model.pt", map_location=torch.device('cpu'))
    check_model(psi_net, wrc_net, hcf_net)


if __name__ == "__main__":
    load_model()
