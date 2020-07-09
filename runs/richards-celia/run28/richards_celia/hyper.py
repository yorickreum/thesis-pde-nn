import json
import os
import subprocess
import time

import torch

import richards_celia
from richards_celia.model import Model


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def objective(trial):
    # Generate the model.
    device = richards_celia.DEVICE
    if torch.cuda.is_available():
        gpu_memory_map = get_gpu_memory_map()
        used_gpus = json.loads((os.getenv("YORICK_USED_GPUS", "[]")))
        unused_gpus = [i for i in gpu_memory_map.keys() if i not in used_gpus]
        if len(unused_gpus) > 0:
            gpu_to_use = min(unused_gpus)
        else:
            gpu_to_use = 0
        used_gpus += [gpu_to_use]
        os.environ["YORICK_USED_GPUS"] = json.dumps(used_gpus)
        device = torch.device('cuda' + ':' + str(gpu_to_use))

    model = Model(
        n_hidden=trial.suggest_int("n_hidden_", 2, 20),
        n_neurons=trial.suggest_int("n_neurons", 8, 64),
        device=device
    )
    parameters = model.h_net.parameters()
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    optimizer = torch.optim.Adam(parameters, lr=lr)

    # main training loop
    startEpoches = time.time()
    for i in range(int(richards_celia.EPOCHS)):
        optimizer.zero_grad()  # clear gradients for next train
        loss = model.loss_func(model.tz_pairs)
        with torch.no_grad():
            if i % richards_celia.LOGSTEPS == 0:
                print("step " + str(i) + ": ")
                print(loss)
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        # check if has converged, break early
        with torch.no_grad():
            if i % 10 == 0:
                if len(model.h_net.losses) >= 11:
                    last_loss_changes = [
                        torch.abs(a_i - b_i)
                        for a_i, b_i in zip(model.h_net.losses[-10:], model.h_net.losses[-11:-1])
                    ]
                    if all(llc <= torch.finfo(richards_celia.DTYPE).eps for llc in last_loss_changes):
                        # or: use max instead of all
                        break
    endEpoches = time.time()

    last_losses = model.h_net.losses[:-100]
    accuracy = sum(last_losses) / len(last_losses)

    return accuracy
