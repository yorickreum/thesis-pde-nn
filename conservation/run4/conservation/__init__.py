import os

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

import torch
from mpmath import mp

# constants
epochs = int(float(os.getenv('EPOCHES', 1e6)))  # parse to float first do deal with '1e6' as str
hidden = int(os.getenv('HIDDEN', 8))
lr = float(os.getenv('LR', .001))
steps = int(os.getenv('STEPS', 32))
dtype = torch.float32
mp.dps = 32  # set number of digits (for pi) @TODO maybe there is a better way than mpmath
pi_tensor = torch.tensor(mp.pi, dtype=dtype)
