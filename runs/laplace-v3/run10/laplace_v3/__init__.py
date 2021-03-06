import os
import torch
from mpmath import mp

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

# constants
EPOCHS = int(float(os.getenv('EPOCHES', 1e4)))  # parse to float first do deal with '1e6' as str
NEURONS = int(os.getenv('NEURONS', 6))
LR = float(os.getenv('LR', .01))
DIVISIONS = int(os.getenv('DIVISIONS', 16))
DTYPE = torch.float64
mp.dps = 32  # set number of digits (for pi) @TODO maybe there is a better way than mpmath
PI_TENSOR = torch.tensor(mp.pi, dtype=DTYPE)
