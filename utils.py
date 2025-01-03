import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Linear, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Sequential, ConvTranspose2d
import scipy.io as sio
from time import time, sleep
import numpy as np
from numpy import cos, sin, sqrt, hypot, arctan2, abs
from numpy.random import rand, randint, randn, uniform
from numpy.linalg import norm, inv
π = np.pi
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set_style("darkgrid")  # adds seaborn style to charts, eg. grid
plt.style.use("dark_background")  # inverts colors to dark theme
plt.rcParams['font.family'] = 'monospace'
np.set_printoptions(precision=3) # set precision for printing numpy arrays
import os
import warnings; warnings.filterwarnings("ignore")
try: 
    JOBID = os.environ["SLURM_JOB_ID"] # get job id from slurm, when training on cluster
    DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # nvidia
    HAS_SCREEN = False # for plotting or saving images
except:
    DEV = torch.device("mps") # apple silicon
    JOBID = "local"
    HAS_SCREEN = True
os.makedirs(f"mg_data/{JOBID}", exist_ok=True)

# DEV = torch.device("cpu") # force cpu

print(f'DEV: {DEV}')

def to_tensor(x, dev=torch.device("cpu")): return torch.tensor(x, dtype=torch.float32, device=dev)

SAVE_DIR = f"mg_data/{JOBID}/lin" # save directory
os.makedirs(SAVE_DIR, exist_ok=True)
# copy the python training to the directory (for cluster) (for local, it fails silently)
os.system(f"cp train.py {SAVE_DIR}/train.py")



######################################################################################################
# dataset
class SXRDataset(Dataset):
    def __init__(self, n, real=False, noise_level:float=0.0, random_remove:int=0, ks=1.0):
        ''' n: number of samples, real: real or simulated data, noise_level: noise level, 
            random_remove: random remove of sxrs, ks: scaling factor for emissivities
        '''
        ds = np.load(f'data/sxr_{"real" if real else "sim"}_ds_{n}.npz')
        self.sxr = to_tensor(np.concatenate([ds['vdi'], ds['vdc'], ds['vde'], ds['hor']], axis=-1), DEV)/ks
        assert self.sxr.shape[-1] == 68, f"wrong sxr shape: {self.sxr.shape}"
        self.em = to_tensor(ds['emiss_lr'], DEV)/ks # emissivities (NxN)
        self.RRH, self.ZZH, self.RRL, self.ZZL = ds['RRH'], ds['ZZH'], ds['RRL'], ds['ZZL'] # grid coordinates
        self.noise_level, self.random_remove, self.ks = noise_level, random_remove, ks
        self.input_size = self.sxr.shape[-1]
        assert len(self.em) == len(self.sxr), f'length mismatch: {len(self.em)} vs {len(self.sxr)}'
    def __len__(self): return len(self.sxr)
    def __getitem__(self, idx):
        x = self.sxr[idx].clone()
        if self.noise_level > 0.0: 
            x = x + torch.randn_like(x) * self.noise_level * x.max()
        if self.random_remove > 0:
            idx_to_remove = torch.randint(0, self.input_size, (self.random_remove,))
            x[idx_to_remove] = 0
        return x, self.em[idx]
    
######################################################################################################
## Network architectures

# custom Swish activation function
class Swish(Module): # swish activation function
    def __init__(self): 
        super(Λ, self).__init__()
        self.β = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
    def forward(self, x): return x*torch.sigmoid(self.β*x)

class Reshape(Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x): return x.view(self.shape)

# activation function
Λ = Swish
# Λ = torch.nn.ReLU # bad
# Λ = torch.nn.Tanh # bad

# architectures
class SXRNetU32(Module): # 32x32
    def __init__(self, input_size, output_size):
        super(SXRNetU32, self).__init__()
        self.enc = Sequential( # encoder
            Linear(input_size, 8), Λ(),
            Linear(8,8), Λ(),
            Linear(8, 8*8), Λ()
        )
        c0, c1, c2, c3 = 2, 3, 4, 5
        self.dec = Sequential( # decoder u-net style 8x8 -> 32x32
            ConvTranspose2d(1, c0, kernel_size=2, stride=2), 
            Conv2d(c0, c1, kernel_size=3, padding=0), Λ(),
            Conv2d(c1, c2, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c2, c3, kernel_size=2, stride=2),
            Conv2d(c3, c3, kernel_size=3, padding=0), Λ(),
            Conv2d(c3, c2, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c2, c1, kernel_size=2, stride=2),
            Conv2d(c1, c0, kernel_size=3, padding=0), Λ(),
            Conv2d(c0, 2, kernel_size=3, padding=0), Λ(),
            Conv2d(2, 1, kernel_size=5, padding=0),
        )
    def forward(self, x):
        x = self.enc(x)
        x = x.view(-1, 1, 8, 8)
        x = self.dec(x)
        return x
    
class SXRNetU32Big(Module): # 32x32
    def __init__(self, input_size, output_size):
        super(SXRNetU32Big, self).__init__()
        self.enc = Sequential( # encoder
            Linear(input_size, 32), Λ(),
            Linear(32,32), Λ(),
            Linear(32, 8*8), Λ()
        )
        c0, c1, c2, c3 = 4, 8, 16, 32
        self.dec = Sequential( # decoder u-net style 8x8 -> 32x32
            ConvTranspose2d(1, c0, kernel_size=2, stride=2), 
            Conv2d(c0, c1, kernel_size=3, padding=0), Λ(),
            Conv2d(c1, c2, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c2, c3, kernel_size=2, stride=2),
            Conv2d(c3, c3, kernel_size=3, padding=0), Λ(),
            Conv2d(c3, c2, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c2, c1, kernel_size=2, stride=2),
            Conv2d(c1, c0, kernel_size=3, padding=0), Λ(),
            Conv2d(c0, 2, kernel_size=3, padding=0), Λ(),
            Conv2d(2, 1, kernel_size=5, padding=0),
        )
    def forward(self, x):
        x = self.enc(x)
        x = x.view(-1, 1, 8, 8)
        x = self.dec(x)
        return x
    
class SXRNetU64(Module): # 32x32
    def __init__(self, input_size, output_size):
        super(SXRNetU64, self).__init__()
        self.enc = Sequential( # encoder
            Linear(input_size, 8), Λ(),
            Linear(8,8), Λ(),
            Linear(8, 8*8), Λ()
        )
        c0, c1, c2, c3 = 2, 3, 4, 5
        self.dec = Sequential( # decoder u-net style 8x8 -> 64x64
            ConvTranspose2d(1, c0, kernel_size=2, stride=2), 
            Conv2d(c0, c1, kernel_size=3, padding=0), Λ(),
            Conv2d(c1, c2, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c2, c3, kernel_size=2, stride=2),
            Conv2d(c3, c3, kernel_size=3, padding=0), Λ(),
            Conv2d(c3, c3, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c3, c2, kernel_size=2, stride=2),
            Conv2d(c2, c2, kernel_size=3, padding=0), Λ(),
            Conv2d(c2, c2, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c2, c1, kernel_size=2, stride=2),
            Conv2d(c1, c0, kernel_size=3, padding=0), Λ(),
            Conv2d(c0, 2, kernel_size=3, padding=0), Λ(),
            Conv2d(2, 1, kernel_size=5, padding=0),
        )
    def forward(self, x):
        x = self.enc(x)
        x = x.view(-1, 1, 8, 8)
        x = self.dec(x)
        return x

class SXRNetLinear1(Module): # 32x32
    def __init__(self, input_size, output_size):
        super(SXRNetLinear1, self).__init__()
        self.net = Sequential(
            Linear(input_size, 128), Λ(),
            Linear(128, output_size*output_size), Λ(),
            Reshape(-1, 1, output_size, output_size)
        )
    def forward(self, x): return self.net(x)

class SXRNetLinear2(Module): # 32x32
    def __init__(self, input_size, output_size):
        super(SXRNetLinear2, self).__init__()
        self.net = Sequential(
            Linear(input_size, 64), Λ(),
            Linear(64, 64), Λ(),
            Linear(64, output_size*output_size), Λ(),
            Reshape(-1, 1, output_size, output_size)
        )
    def forward(self, x): return self.net(x)


######################################################################################################
# math functions
def resize2d(x:np.ndarray, size=(128, 128)):
    if x.shape == size: return x # already the right size
    xt = to_tensor(x).view(1, 1, x.shape[0], x.shape[1])
    xr = torch.nn.functional.interpolate(xt, size=size, mode='bilinear', align_corners=False, antialias=False)
    xrn = xr.numpy().reshape(size)
    return xrn

######################################################################################################
## SXR functions
def calc_sxr(emiss):
    pass

######################################################################################################
# Plotting functions
def plot_net_example(em_ds, em_pred, sxrs:list, rr, zz, titl, labels=['vdi', 'vdc', 'vde', 'hor'], colors=['r','g','b','y']):
    assert len(sxrs) == 4, f"wrong number of sxrs: {len(sxrs)}, should be [vdi, vdc, vde, hor]"
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    bmin, bmax = np.min([em_ds, em_pred]), np.max([em_ds, em_pred]) # min max em
    blevels = np.linspace(bmin, bmax, 13, endpoint=True)
    err_perc = 100*abs(em_ds - em_pred)/(bmax-bmin)
    im00 = axs[0].contourf(rr, zz, em_ds, blevels, cmap="inferno")
    axs[0].set_title("Actual")
    axs[0].set_aspect('equal')
    axs[0].set_ylabel("em")
    fig.colorbar(im00, ax=axs[0]) 
    im01 = axs[1].contourf(rr, zz, em_pred, blevels, cmap="inferno")
    axs[1].set_title("Predicted")
    fig.colorbar(im01, ax=axs[1])
    im02 = axs[2].contour(rr, zz, em_ds, blevels, linestyles='dashed', cmap="inferno")
    axs[2].contour(rr, zz, em_pred, blevels, cmap="inferno")
    axs[2].set_title("Contours")
    fig.colorbar(im02, ax=axs[2])
    im03 = axs[3].contourf(rr, zz, err_perc, cmap="inferno")
    axs[3].set_title("Error %")
    fig.colorbar(im03, ax=axs[3])
    for ax in axs.flatten()[0:3]: ax.grid(False), ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect("equal")
    axs[4].set_title("SXR")
    # plot sxrs
    for sxr, c, l in zip(sxrs, colors, labels): axs[4].plot(sxr, f'{c}s--', label=l)
    axs[4].legend()
    #suptitle
    plt.suptitle(f"SXR Tomography: {titl}")
    plt.tight_layout()
    os.makedirs(f"{SAVE_DIR}/imgs", exist_ok=True)
    plt.show() if HAS_SCREEN else plt.savefig(f"{SAVE_DIR}/imgs/sxr_{titl}.png")
    plt.close()




