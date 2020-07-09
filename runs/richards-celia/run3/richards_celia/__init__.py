import os
import torch
from mpmath import mp

# to make all paths relative to module folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# constants
EPOCHS = int(float(os.getenv('EPOCHES', 1e4)))  # parse to float first do deal with '1e6' as str
LOGSTEPS = int(float(os.getenv('LOGSTEPS', 1)))  # parse to float first do deal with '1e6' as str
HIDDEN = int(os.getenv('HIDDEN', 16))
LR = float(os.getenv('LR', .001))
DIVISIONS = int(os.getenv('DIVISIONS', 256))
# DIVISIONS = int(os.getenv('DIVISIONS', 128))
DTYPE = torch.float32
mp.dps = 32  # set number of digits (for pi) @TODO maybe there is a better way than mpmath
PI_TENSOR = torch.tensor(mp.pi, dtype=DTYPE)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
