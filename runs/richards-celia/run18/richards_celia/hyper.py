import time

import torch

import richards_celia
from richards_celia.model import loss_func, define_model
from richards_celia.model import tz_pairs


def objective(trial):
    # Generate the model.
    # main training loop
    h_net = define_model(neurons=trial.suggest_int("n_neurons", 64, 128))
    parameters = h_net.parameters()
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    optimizer = torch.optim.Adam(parameters, lr=lr)

    startEpoches = time.time()
    for i in range(int(richards_celia.EPOCHS)):
        optimizer.zero_grad()  # clear gradients for next train
        loss = loss_func(h_net, tz_pairs)
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

    last_losses = h_net.losses[:-100]
    accuracy = sum(last_losses) / len(last_losses)

    return accuracy
