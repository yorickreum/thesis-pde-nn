import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
import torch

import richards_celia
from richards_celia.check_net import check_model
from richards_celia.net_hcf import HCFNet
from richards_celia.net_psi import MatricPotentialNet
from richards_celia.net_wrc import WRCNet

print("Device: ", richards_celia.DEVICE)

psi_net = MatricPotentialNet(n_hidden_neurons=richards_celia.HIDDEN)  # schnellste Konvergenz bei 12 hidden
hcf_net = HCFNet(n_hidden_neurons=richards_celia.HIDDEN)  # schnellste Konvergenz bei 12 hidden
wrc_net = WRCNet(n_hidden_neurons=richards_celia.HIDDEN)  # schnellste Konvergenz bei 12 hidden

nets = [psi_net, hcf_net, wrc_net]

if richards_celia.DTYPE == torch.double:
    for n in nets:
        n.double()
psi_net = psi_net.to(device=richards_celia.DEVICE)
hcf_net = hcf_net.to(device=richards_celia.DEVICE)
wrc_net = wrc_net.to(device=richards_celia.DEVICE)

parameters = set()
for n in nets:
    parameters |= set(n.parameters())
optimizer = torch.optim.Adam(parameters, lr=richards_celia.LR)

# data = pandas.read_csv(f"./loam_nod.csv")
data = pandas.read_csv(f"./train_data.csv")
rows = None  # None --> all rows

t = data['t'].values[:rows, None]
# t = data['time'].values[:rows, None]
z = data['z'].values[:rows, None]
# z = data['depth'].values[:rows, None]
tzPairs = np.concatenate((t, z), 1)
tzPairs = torch.tensor(tzPairs,
                       dtype=richards_celia.DTYPE,
                       requires_grad=True,
                       device=richards_celia.DEVICE)

theta = data['theta_train'].values[:rows, None]
# theta = data['theta'].values[:rows, None]
theta = torch.tensor(theta,
                     dtype=richards.DTYPE,
                     requires_grad=False,
                     device=richards.DEVICE)


def loss_func(input: torch.tensor,
              predicted_matrix_potential: torch.tensor):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=richards.DTYPE, device=richards.DEVICE), 1)

    # Check for NaN
    # assert (input == input).any()
    # assert (predicted_matrix_potential == predicted_matrix_potential).any()
    # assert (theta == theta).any()

    predicted_matrix_potential_d = torch.autograd.grad(
        predicted_matrix_potential,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    # predicted_matrix_potential_dt = predicted_matrix_potential_d[:, 0:1]
    predicted_matrix_potential_dz = predicted_matrix_potential_d[:, 1:2]
    predicted_matrix_potential_dzz = torch.autograd.grad(
        predicted_matrix_potential_dz,
        input,
        create_graph=True,
        grad_outputs=ones,
    )
    # predicted_matrix_potential_dzt = (predicted_matrix_potential_dzz[0])[:, 0:1]
    predicted_matrix_potential_dzz = (predicted_matrix_potential_dzz[0])[:, 1:2]

    predicted_matrix_potential_log = torch.log(-predicted_matrix_potential)  # minus cause matrix potential is negative
    # assert (predicted_matrix_potential_log == predicted_matrix_potential_log).any()

    predicted_hcf = hcf_net(predicted_matrix_potential_log)
    # assert (predicted_hcf == predicted_hcf).any()
    predicted_hcf_d = torch.autograd.grad(
        predicted_hcf,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_hcf_dz = predicted_hcf_d[:, 1:2]

    predicted_wrc = wrc_net(predicted_matrix_potential_log)
    # assert (predicted_wrc == predicted_wrc).any()
    predicted_wrc_d = torch.autograd.grad(
        predicted_wrc,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_wrc_dt = predicted_wrc_d[:, 0:1]

    residual = \
        predicted_wrc_dt \
        - predicted_hcf_dz * predicted_matrix_potential_dz \
        - predicted_hcf * predicted_matrix_potential_dzz \
        - predicted_matrix_potential_dz

    # fitting_error = torch.unsqueeze(torch.zeros(len(input), dtype=richards.DTYPE, device=richards.DEVICE), 1)
    fitting_error = predicted_wrc - theta
    # assert (fitting_error == fitting_error).any()

    loss = torch.sum(fitting_error ** 2) + torch.sum(residual ** 2)
    # assert (loss == loss).any()

    return loss


# @TODO train on mini-batches, see https://stackoverflow.com/a/45118712/8666556

# main training loop
startEpoches = time.time()
for i in range(int(richards.EPOCHS)):
    optimizer.zero_grad()  # clear gradients for next train
    prediction = psi_net(tzPairs)
    loss = loss_func(tzPairs, prediction)
    with torch.no_grad():
        psi_net.losses += [loss]
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
    # check if has converged, break early
    with torch.no_grad():
        if i % 10 == 0:
            if len(psi_net.losses) >= 11:
                last_loss_changes = [
                    torch.abs(a_i - b_i)
                    for a_i, b_i in zip(psi_net.losses[-10:], psi_net.losses[-11:-1])
                ]
                if all(llc <= torch.finfo(richards.DTYPE).eps for llc in last_loss_changes):
                    # or: use max instead of all
                    break
endEpoches = time.time()

torch.save(psi_net, "./psi_model.pt")
torch.save(wrc_net, "./wrc_model.pt")
torch.save(hcf_net, "./hcf_model.pt")

print("Runtime of training loop: " + str(endEpoches - startEpoches) + " s")

plt.plot(
    [i for i in range(len(psi_net.losses))],
    [i for i in psi_net.losses]
)
plt.savefig('loss.svg')
plt.savefig('loss.png')
plt.show()

check_model(psi_net, wrc_net, hcf_net)
