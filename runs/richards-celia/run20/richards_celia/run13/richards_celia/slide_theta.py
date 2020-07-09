import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider, Button, RadioButtons

import richards_celia
from richards_celia.func_lib import theta, alpha, theta_s, theta_r, beta


def ts(t, z_space, net_val):
    return net_val


def calc_net_vals(net_h, t, z_top, z_bottom, steps):
    z_lin_space = np.linspace(z_top, z_bottom, steps)
    net_vals = np.zeros(len(z_lin_space))
    for iz in range(len(z_lin_space)):
        z = torch.tensor(z_lin_space[iz], dtype=richards_celia.DTYPE)
        i_net_h = net_h(torch.tensor([t, z], dtype=richards_celia.DTYPE, device=richards_celia.DEVICE))
        net_vals[iz] = i_net_h
    return ts(t, z_lin_space, net_vals)


def calc_theta_vals(net_h_vals):
    return ((alpha * (theta_s - theta_r)) / (alpha + np.abs(net_h_vals) ** beta)) + theta_r


def plot_h(net_h, t, z_top, z_bottom, steps):
    t = torch.tensor(t, dtype=richards_celia.DTYPE, device=richards_celia.DEVICE)
    z_lin_space = np.linspace(z_top, z_bottom, steps)
    net_vals = calc_net_vals(net_h, t, z_top, z_bottom, steps)
    theta_vals = calc_theta_vals(net_vals)
    plot = plt.plot(
        theta_vals,
        z_lin_space
    )
    plt.gca().invert_yaxis()
    plt.title(f'water content')
    plt.xlabel("z / cm")
    plt.ylabel("theta / cm/cm")
    return plot


net_h = torch.load("./h_model.pt", map_location=torch.device('cpu'))
fig, ax = plt.subplots()
# plt.subplots_adjust(left=0.25, bottom=0.25)
z_top = 0
z_bottom = 40
steps = 100
delta_t = 10
t0 = 0
tmax = 900
plot_line, = plot_h(net_h, t0, z_top, z_bottom, 100)
# ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.1, 0.95, 0.8, 0.03], facecolor=axcolor)
stime = Slider(axtime, 'Time', t0, tmax, valinit=t0, valstep=delta_t)


def update(val):
    time = stime.val
    new_net_vals = calc_net_vals(net_h, time, z_top, z_bottom, steps)
    new_theta_vals = calc_theta_vals(new_net_vals)
    plot_line.set_xdata(new_theta_vals)
    ax.relim()
    ax.autoscale_view()
    ax.ticklabel_format(useOffset=False)  # prevent scientific numbers
    fig.canvas.draw_idle()


stime.on_changed(update)


# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    stime.reset()


# button.on_clicked(reset)

# rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)

if __name__ == '__main__':
    plt.show()
