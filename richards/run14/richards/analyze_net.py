import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch

import richards

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def check_net(net):
    data = pandas.read_csv(f"./loam_nod.csv")
    rows = None  # None --> all rows

    tSpace = data['time'].values[:rows, None]
    zSpace = data['depth'].values[:rows, None]

    theta = data['theta'].values[:rows, None]

    t_lin_space = np.linspace(min(tSpace)[0], max(tSpace)[0], 100)
    z_lin_space = np.linspace(min(zSpace)[0], max(zSpace)[0], 100)

    net_vals = np.zeros((len(z_lin_space), len(t_lin_space)))
    for it in range(len(t_lin_space)):
        for iz in range(len(z_lin_space)):
            t = torch.tensor(t_lin_space[it], dtype=richards.DTYPE)
            z = torch.tensor(z_lin_space[iz], dtype=richards.DTYPE)
            net_tz = net(torch.tensor([t, z], dtype=richards.DTYPE, device=richards.DEVICE)).cpu()
            net_vals[iz][it] = net_tz

    levels = np.linspace(net_vals.min(), net_vals.max(), 100)
    cs = plt.contourf(
        t_lin_space,
        z_lin_space,
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


net = torch.load(f"./model.pt", map_location=torch.device('cpu'))
net.eval()

check_net(net)
