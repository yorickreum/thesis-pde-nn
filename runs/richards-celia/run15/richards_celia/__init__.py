import os
import torch
import numpy as np

# from mpmath import mp

# to make all paths relative to module folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# constants
EPOCHS = int(float(os.getenv('EPOCHES', 100)))  # parse to float first do deal with '1e6' as str
LOGSTEPS = int(float(os.getenv('LOGSTEPS', 1)))  # parse to float first do deal with '1e6' as str
HIDDEN = int(os.getenv('HIDDEN', 64))
LR = float(os.getenv('LR', .005))
DIVISIONS = int(os.getenv('DIVISIONS', 100))
# DIVISIONS = int(os.getenv('DIVISIONS', 128))
DTYPE = torch.float32
# mp.dps = 32  # set number of digits (for pi) @TODO maybe there is a better way than mpmath
# PI_TENSOR = torch.tensor(mp.pi, dtype=DTYPE)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# reproducibility
torch.manual_seed(0)
np.random.seed(0)
