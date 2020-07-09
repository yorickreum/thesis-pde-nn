import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import richards_celia
from richards_celia.check_net import plot_h
from richards_celia.func_lib import theta, K
from richards_celia.net_h import PressureHeadNet

print("Device: ", richards_celia.DEVICE)

h_net = PressureHeadNet(n_hidden_neurons=richards_celia.HIDDEN)  # schnellste Konvergenz bei 12 hidden

nets = [h_net]

if richards_celia.DTYPE == torch.double:
    for n in nets:
        n.double()
h_net = h_net.to(device=richards_celia.DEVICE)

parameters = set()
for n in nets:
    parameters |= set(n.parameters())
optimizer = torch.optim.Adam(parameters, lr=richards_celia.LR)

t = np.linspace(0, 720, richards_celia.DIVISIONS)
z = np.linspace(-10, -70, richards_celia.DIVISIONS)

# TODO okay? enforce time boundary only weakly, (ct for ct in t if ct > 0)
tz_pairs_top_boundary = \
    torch.tensor([[it, 40] for it in (ct for ct in t if ct > 10)],
                 dtype=richards_celia.DTYPE, device=richards_celia.DEVICE)
h_top_boundary = -20.7
tz_pairs_bottom_boundary = \
    torch.tensor([[it, 0] for it in (ct for ct in t if ct > 10)],
                 dtype=richards_celia.DTYPE, device=richards_celia.DEVICE)
h_bottom_boundary = -61.5

tz_pairs_initial_boundary = \
    torch.tensor([[0, iz] for iz in z],
                 dtype=richards_celia.DTYPE, device=richards_celia.DEVICE)
h_initial_boundary = -61.5


# def trial(input):
#     return (-61.5) + input[:, 0:1] * h_net(input)


def loss_func(input: torch.tensor):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=richards_celia.DTYPE, device=richards_celia.DEVICE), 1)

    # predicted_h = h_net(tz_pairs)
    predicted_h_trial = h_net(input)
    predicted_h_initial_boundary = h_net(tz_pairs_initial_boundary)
    predicted_h_bottom_boundary = h_net(tz_pairs_bottom_boundary)
    predicted_h_top_boundary = h_net(tz_pairs_top_boundary)

    predicted_h_d = torch.autograd.grad(
        predicted_h_trial,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dz = predicted_h_d[:, 1:2]

    predicted_theta = theta(predicted_h_trial)
    predicted_K = K(predicted_h_trial)

    predicted_theta_d = torch.autograd.grad(
        predicted_theta,
        input,
        create_graph=True,
        grad_outputs=ones,
    )[0]
    predicted_theta_dt = predicted_theta_d[:, 0:1]

    predicted_second_term = predicted_K * predicted_h_dz
    predicted_second_term_d = torch.autograd.grad(
        predicted_second_term,
        input,
        create_graph=True,
        grad_outputs=ones,
    )[0]
    predicted_second_term_dz = predicted_second_term_d[:, 1:2]

    predicted_K_d = torch.autograd.grad(
        predicted_K,
        input,
        create_graph=True,
        grad_outputs=ones,
    )[0]
    predicted_K_dz = predicted_K_d[:, 1:2]

    residual = predicted_theta_dt - predicted_second_term_dz - predicted_K_dz

    residual_h_initial_boundary = predicted_h_initial_boundary - h_initial_boundary
    residual_h_bottom_boundary = predicted_h_bottom_boundary - h_bottom_boundary
    residual_h_top_boundary = predicted_h_top_boundary - h_top_boundary

    # fitting_error = torch.unsqueeze(torch.zeros(len(input), dtype=richards_celia.DTYPE, device=richards_celia.DEVICE),1)
    # fitting_error = predicted_wrc - theta
    # assert (fitting_error == fitting_error).any()
    #  + torch.sum(fitting_error ** 2) \

    loss = torch.sum(residual ** 2) / len(residual) \
           + torch.sum(residual_h_initial_boundary ** 2) / len(residual_h_initial_boundary) \
           + torch.sum(residual_h_bottom_boundary ** 2) / len(residual_h_bottom_boundary) \
           + torch.sum(residual_h_top_boundary ** 2) / len(residual_h_top_boundary)
    # assert (loss == loss).any()

    return loss


# @TODO train on mini-batches, see https://stackoverflow.com/a/45118712/8666556

tz_pairs = []
for r in itertools.product(t, z):
    tz_pairs += [[r[0], r[1]]]
tz_pairs = torch.tensor(tz_pairs, dtype=richards_celia.DTYPE, requires_grad=True, device=richards_celia.DEVICE)

# main training loop
startEpoches = time.time()
for i in range(int(richards_celia.EPOCHS)):
    optimizer.zero_grad()  # clear gradients for next train
    loss = loss_func(tz_pairs)
    with torch.no_grad():
        h_net.losses += [loss]
        if i % richards_celia.LOGSTEPS == 0:
            print("step " + str(i) + ": ")
            print(loss)
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    # check if has converged, break early
    with torch.no_grad():
        if i % 10 == 0:
            if len(h_net.losses) >= 11:
                last_loss_changes = [
                    torch.abs(a_i - b_i)
                    for a_i, b_i in zip(h_net.losses[-10:], h_net.losses[-11:-1])
                ]
                if all(llc <= torch.finfo(richards_celia.DTYPE).eps for llc in last_loss_changes):
                    # or: use max instead of all
                    break
endEpoches = time.time()

torch.save(h_net, "./h_model.pt")

print("Runtime of training loop: " + str(endEpoches - startEpoches) + " s")

plt.plot(
    [i for i in range(len(h_net.losses))],
    [i for i in h_net.losses]
)
plt.savefig('loss.svg')
plt.savefig('loss.png')
plt.show()

plot_h(h_net, 0, 0, 40, 100)
