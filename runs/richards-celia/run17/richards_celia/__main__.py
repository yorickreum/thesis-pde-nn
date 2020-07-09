import time

import matplotlib.pyplot as plt
import optuna
import torch

import richards_celia
from richards_celia.hyper import objective
from richards_celia.model import loss_func, tz_pairs, define_model


def train():
    # main training loop
    h_net = define_model(neurons=richards_celia.HIDDEN)
    parameters = h_net.parameters()
    optimizer = torch.optim.Adam(parameters, lr=richards_celia.LR)

    startEpoches = time.time()
    for i in range(int(richards_celia.EPOCHS)):
        optimizer.zero_grad()  # clear gradients for next train
        loss = loss_func(h_net=h_net, input=tz_pairs)
        with torch.no_grad():
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
        [i for i in h_net.losses],
        label="total loss"
    )
    plt.plot(
        [i for i in range(len(h_net.pde_loss))],
        [i for i in h_net.pde_loss],
        label="pde loss"
    )
    plt.plot(
        [i for i in range(len(h_net.bc_initial_losses))],
        [i for i in h_net.bc_initial_losses],
        label="bc initial loss"
    )
    plt.plot(
        [i for i in range(len(h_net.bc_top_losses))],
        [i for i in h_net.bc_top_losses],
        label="bc top loss"
    )
    plt.plot(
        [i for i in range(len(h_net.bc_bottom_losses))],
        [i for i in h_net.bc_bottom_losses],
        label="bc bottom loss"
    )
    plt.legend()
    plt.savefig('loss.pdf')
    plt.savefig('loss.png')
    plt.show()

    # plot_h(h_net, 0, 0, 40, 100)
    # plot_h(h_net, 360, 0, 40, 100)


def hyperstudy():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=richards_celia.HYPERTRIALS)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


hyperstudy()
