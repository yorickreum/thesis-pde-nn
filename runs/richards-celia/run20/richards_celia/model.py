import itertools

import torch

import richards_celia
from richards_celia.func_lib import theta, K
from richards_celia.net_h import PressureHeadNet

import numpy as np

print("Device: ", richards_celia.DEVICE)


def define_model(n_hidden, n_neurons):
    h_net = PressureHeadNet(n_hidden_layers=n_hidden, n_hidden_neurons=n_neurons)  # schnellste Konvergenz bei 12 hidden
    if richards_celia.DTYPE == torch.double:
        h_net.double()
    h_net = h_net.to(device=richards_celia.DEVICE)
    return h_net


t = np.linspace(0, 720, richards_celia.DIVISIONS)
z = np.linspace(0, 40, richards_celia.DIVISIONS)

# TODO okay? enforce time boundary only weakly, (ct for ct in t if ct > 0)
tz_pairs_top_boundary = \
    torch.tensor([[it, 40] for it in (ct for ct in t if ct >= 0)],
                 dtype=richards_celia.DTYPE, device=richards_celia.DEVICE,
                 requires_grad=True)
# h_top_boundary = -20.7
# q_top_boundary = 13.69 / (60 * 60)
h_top_boundary = \
    torch.tensor([[-20.7 - 40.8 * np.exp(-1e6 * it)] for it in (ct for ct in t if ct >= 0)],
                 dtype=richards_celia.DTYPE, device=richards_celia.DEVICE)

tz_pairs_bottom_boundary = \
    torch.tensor([[it, 0] for it in (ct for ct in t if ct >= 0)],
                 dtype=richards_celia.DTYPE, device=richards_celia.DEVICE,
                 requires_grad=True)
h_bottom_boundary = -61.5

tz_pairs_initial_boundary = \
    torch.tensor([[0, iz] for iz in z],
                 dtype=richards_celia.DTYPE, device=richards_celia.DEVICE,
                 requires_grad=True)
h_initial_boundary = -61.5

# @TODO entferne top und bottom aus tzPairs
tz_pairs = []
for r in itertools.product(t, z):
    pair = [r[0], r[1]]
    # if not (r[0] == 0 or r[1] == 0 or r[1] == 40):
    tz_pairs += [pair]
tz_pairs = torch.tensor(tz_pairs, dtype=richards_celia.DTYPE, requires_grad=True, device=richards_celia.DEVICE)


# h_initial_boundary = torch.tensor(
#     [[i] for i in np.linspace(
#         h_top_boundary,
#         h_bottom_boundary,
#         richards_celia.DIVISIONS)
#      ],
#     dtype=richards_celia.DTYPE,
#     device=richards_celia.DEVICE,
#     requires_grad=False
# )


# def trial(input):
#     return (-61.5) + input[:, 0:1] * h_net(input)

# def ts(input: torch.Tensor):
#     # t = input[:, 0:1]
#     z = input[:, 1:2]
#     net_val = h_net(input)
#     return z * (z - 40) * net_val \
#            - 61.5 * (z - 40) * (-1 / 40) \
#            - 20.7 * z * (1 / 40)


def loss_func(h_net, input: torch.Tensor):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=richards_celia.DTYPE, device=richards_celia.DEVICE), 1)

    predicted_h_trial = h_net(tz_pairs)  # ts(input)
    predicted_h_initial_boundary = h_net(tz_pairs_initial_boundary)  # ts(tz_pairs_initial_boundary)
    predicted_h_bottom_boundary = h_net(tz_pairs_bottom_boundary)
    predicted_h_top_boundary = h_net(tz_pairs_top_boundary)

    # predicted_h_top_boundary_d = torch.autograd.grad(
    #     predicted_h_top_boundary,
    #     tz_pairs_top_boundary,
    #     create_graph=True,
    #     grad_outputs=torch.unsqueeze(
    #         torch.ones(len(tz_pairs_top_boundary), dtype=richards_celia.DTYPE, device=richards_celia.DEVICE), 1)
    # )[0]
    # predicted_h_top_boundary_dz = predicted_h_top_boundary_d[:, 1:2]
    #
    # predicted_q_top_boundary = K(predicted_h_top_boundary) * (1 - predicted_h_top_boundary_dz)

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

    # @TODO check the signs here
    residual = predicted_theta_dt - predicted_second_term_dz - predicted_K_dz

    residual_h_initial_boundary = predicted_h_initial_boundary - h_initial_boundary
    residual_h_bottom_boundary = predicted_h_bottom_boundary - h_bottom_boundary
    residual_h_top_boundary = predicted_h_top_boundary - h_top_boundary
    # residual_q_top_boundary = predicted_q_top_boundary - q_top_boundary

    # fitting_error = torch.unsqueeze(torch.zeros(len(input), dtype=richards_celia.DTYPE, device=richards_celia.DEVICE),1)
    # fitting_error = predicted_wrc - theta
    # assert (fitting_error == fitting_error).any()
    #  + torch.sum(fitting_error ** 2) \

    pde_res = torch.sum(residual ** 2) / len(residual)
    boundary_bottom_res = torch.sum(residual_h_bottom_boundary ** 2) / len(residual_h_bottom_boundary)
    boundary_top_res = torch.sum(residual_h_top_boundary ** 2) / len(residual_h_top_boundary)
    # boundary_top_res = torch.sum(residual_q_top_boundary ** 2) / len(residual_q_top_boundary)
    boundary_initial_res = torch.sum(residual_h_initial_boundary ** 2) / len(residual_h_initial_boundary)
    print("pde_res: " + str(pde_res))
    print("boundary_bottom_res: " + str(boundary_bottom_res))
    print("boundary_top_res: " + str(boundary_top_res))
    print("boundary_initial_res: " + str(boundary_initial_res))
    loss = pde_res + (boundary_bottom_res + boundary_top_res + boundary_initial_res)
    # loss = pde_res + boundary_initial_res
    # assert (loss == loss).any()
    h_net.pde_loss += [pde_res]
    h_net.bc_initial_losses += [boundary_initial_res]
    h_net.bc_top_losses += [boundary_top_res]
    h_net.bc_bottom_losses += [boundary_bottom_res]
    h_net.losses += [loss]

    return loss

# @TODO train on mini-batches, see https://stackoverflow.com/a/45118712/8666556
