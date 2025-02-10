from common import * # import all the common parameters

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Linear, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Sequential, ConvTranspose2d
import scipy.io as sio
from time import time, sleep
import numpy as np
from numpy import cos, sin, sqrt, hypot, arctan2, abs, log
from numpy.random import rand, randint, randn, uniform
from numpy.linalg import norm, inv
π = np.pi
from tqdm import tqdm

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")  # adds seaborn style to charts, eg. grid
plt.style.use("dark_background")  # inverts colors to dark theme
plt.rcParams['font.family'] = 'monospace' 
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.2
CMAP_NAME = "inferno" #'plasma' # colormap
CMAP = plt.get_cmap(CMAP_NAME)
plt.rcParams['image.cmap'] = CMAP_NAME
np.set_printoptions(precision=3) # set precision for printing numpy arrays

import os
import warnings; warnings.filterwarnings("ignore")
try: 
    JOBID = os.environ["SLURM_JOB_ID"] # get job id from slurm, when training on cluster
    DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # nvidia
    HAS_SCREEN = False # for plotting or saving images
except:
    JOBID = "local"
    DEV = torch.device("mps") # apple silicon
    HAS_SCREEN = True
os.makedirs(f"mg_data/{JOBID}", exist_ok=True)

DEV = torch.device("cpu") # force cpu

print(f'DEV: {DEV}')

def to_tensor(x, dev=torch.device("cpu")): return torch.tensor(x, dtype=torch.float32, device=dev)

SAVE_DIR = f"mg_data/{JOBID}/lin" # save directory
os.makedirs(SAVE_DIR, exist_ok=True)
# copy the python training to the directory (for cluster) (for local, it fails silently)
os.system(f"cp train.py {SAVE_DIR}/train.py")

# set random seeds
np.random.seed(42)
torch.manual_seed(42)

######################################################################################################
# dataset
class SXRDataset(Dataset):
    def __init__(self, n, gs=GSIZE, real=False, noise_level:float=0.0, random_remove:int=0, calc_sxr=False, rescale=True):
        ''' n: number of samples, gs: grid_size, real: real or simulated data, 
            noise_level: noise level, random_remove: random remove of sxrs, 
            rescale: rescale the emissivities and sxr so that max em == 1, 
            calc_sxr: recalculate the SXR from the emissivities
        '''
        self.noise_level, self.random_remove, self.gs = noise_level, random_remove, gs
        ds = np.load(f'data/sxr_{"real" if real else "sim"}_ds_gs{gs}_n{n}.npz')
        self.em = to_tensor(ds['emiss'], DEV).view(-1,gs,gs) # emissivities (nxNxN)
        self.or_sxr = to_tensor(np.concatenate([ds['vdi'], ds['vdc'], ds['vde'], ds['hor']], axis=-1), DEV) # sxr from data
        if calc_sxr: # recalculate the SXR from the emissivities
            ems = self.em.cpu().numpy() # emissivities
            to_conc = [eval_rfx_sxrs(create_default_rfx_fan(n), ems) for n in RFX_SXR_NAMES]
            self.sxr = to_tensor(np.concatenate(to_conc, axis=-1), DEV)
        else: self.sxr = self.or_sxr.clone() # use the original SXR
        if rescale: # rescale the emissivities and sxr so that max em == 1
            self.scales = (self.em.view(-1, gs*gs).max(axis=1).values).view(-1, 1)
            self.em = (self.em.view(-1, gs*gs)/self.scales).view(-1, gs, gs)
            self.sxr /= self.scales # rescale the sxr
            # #normalize (works only on scaled data)
            # d = np.load(f'data/rfx_sxr_means_stds_{gs}.npz')
            # μs, Σs = to_tensor(d['means'], DEV), to_tensor(d['stds'], DEV)
            # assert len(μs) == len(Σs) == self.sxr.shape[-1], f"wrong means or stds shape: {len(μs)} vs {len(Σs)} vs {self.sxr.shape[-1]}"
            # # self.sxr = (self.sxr - μs)/Σs # normalize the sxr
            # self.sxr = self.sxr - μs # remove the mean

        self.input_size = self.sxr.shape[-1]
        assert self.sxr.shape[-1] == 68, f"wrong sxr shape: {self.sxr.shape}"
        self.RR, self.ZZ = ds['RR'], ds['ZZ'] # grid coordinates
        assert len(self.em) == len(self.sxr), f'length mismatch: {len(self.em)} vs {len(self.sxr)}'
    def __len__(self): return len(self.sxr)
    def __getitem__(self, idx):
        x = self.sxr[idx].clone()
        if self.noise_level > 0.0: 
            x = x + torch.randn_like(x) * self.noise_level * x.max()
        if self.random_remove > 0:
            n_to_remove = torch.randint(1, self.random_remove, (1,))
            idx_to_remove = torch.randint(0, self.input_size, (n_to_remove,))
            x[idx_to_remove] = 0
        return x, self.em[idx]
    def show_examples(self, n_plot=10):
        fig, axs = plt.subplots(2, n_plot, figsize=(3*n_plot, 5))
        np.random.seed(42)
        idxs = np.random.randint(0, len(self), n_plot)
        for i, j in enumerate(idxs):
            sxr, em = self[j][0].cpu().numpy().squeeze(), self[j][1].cpu().numpy()
            axs[0,i].contourf(self.RR, self.ZZ, em, 100, cmap="inferno")
            axs[0,i].axis("off")
            axs[0,i].set_aspect("equal")
            fig.colorbar(axs[0,i].contourf(self.RR, self.ZZ, em, 100, cmap="inferno"), ax=axs[0,i])
            #plot sxr
            axs[1,i].plot(sxr, 'rs')
        plt.show() if HAS_SCREEN else plt.savefig(f"mg_data/{JOBID}/dataset.png")
        plt.close()
        return idxs
######################################################################################################
## Network architectures

# custom Swish activation function
class Swish(Module): # swish activation functio
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
# Λ = torch.nn.Sigmoid

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
        x = x.view(-1, 32, 32)
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
        x = x.view(-1, 32, 32)
        return x
    
class SXRNetU55(Module): # 55x55
    def __init__(self, input_size, output_size):
        super(SXRNetU55, self).__init__()
        self.enc = Sequential( # encoder
            Linear(input_size, 32), Λ(),
            Linear(32,32), Λ(),
            Linear(32, 16*16), Λ()
        )
        c0, c1, c2, c3 = 4, 8, 16, 32
        self.dec = Sequential( # decoder u-net style 16x16 -> 55x55
            ConvTranspose2d(1, c0, kernel_size=2, stride=2), 
            Conv2d(c0, c0, kernel_size=3, padding=1), Λ(),
            # Conv2d(c0, c0, kernel_size=3, padding=0), Λ(),
            Conv2d(c0, c0, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c0, c1, kernel_size=2, stride=2),
            Conv2d(c1, c1, kernel_size=3, padding=1), Λ(),
            # Conv2d(c1, c1, kernel_size=3, padding=0), Λ(),
            # Conv2d(c1, c1, kernel_size=3, padding=0), Λ(),
            # ConvTranspose2d(c1, c2, kernel_size=2, stride=2),
            Conv2d(c1, c2, kernel_size=3, padding=0), Λ(),
            # Conv2d(c2, c2, kernel_size=3, padding=0), Λ(),
            Conv2d(c2, 1, kernel_size=4, padding=0), Λ(),
            # ConvTranspose2d(c2, c1, kernel_size=2, stride=2),
            # Conv2d(c1, c1, kernel_size=3, padding=0), Λ(),
            # Conv2d(c1, c1, kernel_size=3, padding=0), Λ(),
            # Conv2d(c1, c0, kernel_size=3, padding=0), Λ(),
            # Conv2d(c3, 1, kernel_size=4, padding=0),
        )
    def forward(self, x):
        x = self.enc(x)
        x = x.view(-1, 1, 16, 16)
        x = self.dec(x)
        x = x.view(-1, 55, 55)
        return x
    
class SXRNetU110(Module): # 55x55
    def __init__(self, input_size, output_size):
        super(SXRNetU110, self).__init__()
        self.enc = Sequential( # encoder
            Linear(input_size, 32), Λ(),
            Linear(32,32), Λ(),
            Linear(32, 16*16), Λ()
        )
        c0, c1, c2, c3 = 4, 8, 16, 32
        self.dec = Sequential( # decoder u-net style 16x16 -> 110x110
            ConvTranspose2d(1, c0, kernel_size=2, stride=2), 
            Conv2d(c0, c0, kernel_size=3, padding=0), Λ(),
            # Conv2d(c0, c0, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c0, c1, kernel_size=2, stride=2),
            Conv2d(c1, c1, kernel_size=3, padding=0), Λ(),
            # Conv2d(c1, c1, kernel_size=3, padding=0), Λ(),
            ConvTranspose2d(c1, c2, kernel_size=2, stride=2),
            Conv2d(c2, c2, kernel_size=3, padding=0), Λ(),
            Conv2d(c2, c1, kernel_size=3, padding=0), Λ(),
            Conv2d(c1, 1, kernel_size=3, padding=0), Λ(),
            # Conv2d(c1, 1, kernel_size=3, padding=0), Λ(),

        )
    def forward(self, x):
        x = self.enc(x)
        x = x.view(-1, 1, 16, 16)
        x = self.dec(x)
        x = x.view(-1, 110, 110)
        return x
    
class SXRNetU64(Module): # 64x64
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
        x = x.view(-1, 64, 64)
        return x

class SXRNetLinear1(Module): 
    def __init__(self, input_size, output_size):
        super(SXRNetLinear1, self).__init__()
        self.net = Sequential(
            Linear(input_size, 128), Λ(),
            Linear(128, output_size*output_size), Λ(),
            Reshape(-1, output_size, output_size)
        )
    def forward(self, x): return self.net(x)

class SXRNetLinear2(Module): 
    def __init__(self, input_size, output_size):
        super(SXRNetLinear2, self).__init__()
        self.net = Sequential(
            Linear(input_size, 128), Λ(),
            Linear(128, 128), Λ(),
            Linear(128, output_size*output_size), Λ(),
            Reshape(-1, output_size, output_size)
        )
    def forward(self, x): return self.net(x)

class SXRNetLinCos(Module):
    def __init__(self, input_size, output_size, latent_size=16):
        super(SXRNetLinCos, self).__init__()
        self.enc = Sequential( # encoder
            Linear(input_size, 64), Λ(),
            # Linear(128, 128), Λ(),
            Linear(64, latent_size), Λ()
        )
        self.dec = Sequential( # decoder
            Linear(latent_size, 64), Λ(),
            # Linear(128, 128), Λ(),
            Linear(64, output_size*output_size), Λ(),
            Reshape(-1, output_size, output_size)
        )
    def forward(self, x): 
        x = self.enc(x)
        x = x/torch.norm(x, dim=-1, keepdim=True) # normalize
        x = self.dec(x)
        return x

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
# helper functions
def wrap_angle(α): return np.arctan2(np.sin(α), np.cos(α))

def gaussian(v, μ=np.array([R0+L/2, Z0+L/2]), Σ=np.array([[L/4,0],[0,L/4]]), polar=False):
    rshape = v.shape[:-1] # save the original shape
    v = v.reshape(-1, 2) # flatten the input to 2D
    d = v-μ # difference vector
    if polar: d[:,1] = wrap_angle(d[:,1]) 
    g = np.exp(-0.5*np.sum(d @ inv(Σ) * d, axis=-1)) # gaussian formula
    r = g.reshape(rshape) # return the result in the original shape
    return r

def create_line(c, θ, n=20*GSIZE): # create a line from a center and an absolute angle
    cθ, sθ = cos(θ), sin(θ)
    if np.abs(cθ) > np.abs(sθ): # less than 45 degrees
        x = np.linspace(R0-GSPAC/2, R1+GSPAC/2, n)
        y = (sθ/cθ)*x + (c[1] - (sθ/cθ)*c[0])
    else: # more than 45 degrees
        y = np.linspace(Z0-GSPAC/2, Z1+GSPAC/2, n)
        x = (cθ/sθ)*y + (c[0] - (cθ/sθ)*c[1])
    # keep only points inside the grid/first wall
    # idxs = (R0-GSPAC/2 <= x) & (x <= R1+GSPAC/2) & (Z0-GSPAC/2 <= y) & (y <= Z1+GSPAC/2) # inside grid
    idxs = (x-RM)**2 + (y-ZM)**2 <= (R_FW+GSPAC/2)**2 # inside first wall #TODO: not exact Radius FW < L/2
    # if sum(idxs) == 0: print(f'Warning: line outside: c={c}, θ={θ:.2f}')
    return np.stack((x[idxs], y[idxs]), axis=-1)


def line_mask(c, θn, θl, n=3*GSIZE): # mask for a line, c=center, θn=normal angle, θl=angle of the line wrt the normal
    lin = create_line(c, θn+θl, n)
    mask = np.zeros((GSIZE, GSIZE))
    for l in lin:
        rl, zl = l # line point
        ir1, ir2 = np.argsort((R-rl)**2)[:2] # closest idxs in R
        iz1, iz2 = np.argsort((Z-zl)**2)[:2] # closest idxs in Z
        dr1, dr2 = (R[ir1]-rl)**2, (R[ir2]-rl)**2 # distances
        dz1, dz2 = (Z[iz1]-zl)**2, (Z[iz2]-zl)**2 # distances
        wr2, wz2, wr1, wz1 = dr1/(dr1+dr2), dz1/(dz1+dz2), dr2/(dr1+dr2), dz2/(dz1+dz2) # weights
        w11,w12,w21,w22 = wr1*wz1, wr1*wz2, wr2*wz1, wr2*wz2
        assert np.isclose(w11+w12+w21+w22, 1), f'{w11:.2f}, {w12:.2f}, {w21:.2f}, {w22:.2f}, sum={w11+w12+w21+w22:.5f}'
        mask[iz1, ir1] += w11*GSIZE/n
        mask[iz1, ir2] += w12*GSIZE/n
        mask[iz2, ir1] += w21*GSIZE/n
        mask[iz2, ir2] += w22*GSIZE/n

    # plt.figure(figsize=(8,8))
    # plt.scatter(RR, ZZ, c=mask, s=8)
    # plt.plot(lin[:,0], lin[:,1], 'r')
    # plt.plot(FW[:,0], FW[:,1], 'w')
    # plt.axis('equal')
    # plt.colorbar()
    # plt.show()
    # plt.close()

    mask = mask.reshape(-1)
    mask_idxs = np.where(mask > 0)[0]
    mask = mask[mask_idxs]
    mask *= cos(θl)#**2 # the inclination of the line reduces the mask by cos(θl) #TODO: check if this is correct
    mask *= GSPAC # multiply by the grid spacing
    return mask, mask_idxs

# function to create a RFX fan of rays
def create_rfx_fan(nrays, start_angle, span_angle, pinhole_position, idxs_to_keep, ret_all=False):
    α_normal = start_angle+span_angle/2 # normal incidence angle of the fan
    αs_rays = np.linspace(start_angle, start_angle+span_angle, nrays) # absolutre angles of the rays
    αs_rays = αs_rays[idxs_to_keep] # keep only the rays specified by idxs_to_keep
    αs_incidence = αs_rays - α_normal # angles of incidence wrt the normal
    rays = [create_line(pinhole_position, αr) for αr in αs_rays]
    fan = [line_mask(pinhole_position, α_normal, αi) for αi in αs_incidence]
    if ret_all: return rays, fan, αs_incidence
    else: return fan

def create_default_rfx_fan(name='VDI', ret_all=False):
    assert name in RFX_SXR_NAMES, f"wrong fan name: {name}, should be in {RFX_SXR_NAMES}"
    i = RFX_SXR_NAMES_2_IDXS[name]
    return create_rfx_fan(RFX_SXR_NRAYS[i], RFX_SXR_STARTS[i], RFX_SXR_SPANS[i], RFX_SXR_PINHOLES[i], RFX_SXR_TO_KEEP[i], ret_all)


def eval_rfx_sxr(fan, emiss):
    ''' fan: fan of rays [(mask, mask_idxs), ...] emiss: emissivity distribution (gs x gs) 
        evaluate the SXR for a fan of rays
    '''
    return eval_rfx_sxrs(fan, emiss.reshape(1, GSIZE, GSIZE)).reshape(len(fan))

def eval_rfx_sxrs(fan, emisss:np.ndarray):
    ''' fan: fan of rays [(mask, mask_idxs), ...] emisss: emissivity distributions (n x gs x gs) 
        evaluate the SXR for a fan of rays for a batch of emissivities
    '''
    assert emisss.shape[1:] == (GSIZE, GSIZE), f"wrong emissivity shape: {emisss.shape}, should be {(GSIZE, GSIZE)}"
    sxrs = np.zeros((len(emisss), len(fan)))
    for i, (m, mi) in enumerate(fan): # for each ray, m: mask, mi: mask indexes
        ems_masked = emisss.reshape(len(emisss), GSIZE*GSIZE)[:,mi] # mask the emissivity
        sxrs[:,i] = np.sum(ems_masked*m, axis=-1)
    return sxrs

class RFX_SXR():
    def __init__(self, gs=GSIZE, emiss=None): # gs: grid size
        self.gs = gs
        self.fan_names = ['vdi', 'vdc', 'vde', 'hor']
        self.rfx_sxr = {self.fan_names[i]:create_rfx_fan(RFX_SXR_NRAYS[i], RFX_SXR_STARTS[i], RFX_SXR_SPANS[i], RFX_SXR_PINHOLES[i], RFX_SXR_TO_KEEP[i]) for i in range(4)}
        self.vdi_n = len(self.rfx_sxr['vdi'])
        self.vdc_n = len(self.rfx_sxr['vdc'])
        self.vde_n = len(self.rfx_sxr['vde'])
        self.hor_n = len(self.rfx_sxr['hor'])
        if emiss is not None: return self.eval_rfx(emiss)
    def eval_on_fan(self, emiss, fan_name='vdi'):
        ''' emiss: emissivity distribution (gs x gs) 
            fan: vdi, vdc, vde, hor
        '''
        assert fan_name in self.fan_names, f"wrong fan name: {fan_name}"
        assert emiss.shape == (self.gs, self.gs), f"wrong emissivity shape: {emiss.shape}, should be {(self.gs, self.gs)}"
        f = self.rfx_sxr[fan_name]
        return eval_rfx_sxr(f, emiss)
    def eval_rfx(self, emiss):
        ''' emiss: emissivity distribution (gs x gs) 
            evaluate on all the fans
        '''
        sxr = np.zeros(self.vdi_n+self.vdc_n+self.vde_n+self.hor_n)
        cum_idx = 0 # cumulative index
        for fan in self.fan_names:
            ni = len(self.rfx_sxr[fan])
            sxr[cum_idx:cum_idx+ni] = self.eval_on_fan(emiss, fan)
            cum_idx += ni
        return sxr



######################################################################################################
# Plotting functions
def plot_net_example(em_ds, em_pred, sxrs:list, rr, zz, titl, labels=['vdi', 'vdc', 'vde', 'hor'], colors=['r','g','b','y']):
    assert len(sxrs) == 4, f"wrong number of sxrs: {len(sxrs)}, should be [vdi, vdc, vde, hor]"
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    bmin, bmax = np.min([em_ds, em_pred]), np.max([em_ds, em_pred]) # min max em
    blevels = np.linspace(bmin, bmax, 13, endpoint=True)
    err_perc = 100*abs(em_ds - em_pred)/(bmax-bmin)
    im00 = axs[0].contourf(rr, zz, em_ds, blevels, cmap="inferno")
    #plot FW
    axs[0].plot(FW[:,0], FW[:,1], 'w', lw=1)
    axs[0].set_title("Actual")
    axs[0].set_aspect('equal')
    axs[0].set_ylabel("em")
    fig.colorbar(im00, ax=axs[0]) 
    im01 = axs[1].contourf(rr, zz, em_pred, blevels, cmap="inferno")
    axs[1].plot(FW[:,0], FW[:,1], 'w', lw=1)
    axs[1].set_aspect('equal')
    axs[1].set_title("Predicted")
    fig.colorbar(im01, ax=axs[1])
    im02 = axs[2].contour(rr, zz, em_ds, blevels, linestyles='dashed', cmap="inferno")
    axs[2].contour(rr, zz, em_pred, blevels, cmap="inferno")
    axs[2].plot(FW[:,0], FW[:,1], 'w', lw=1)
    axs[2].set_aspect('equal')
    axs[2].set_title("Contours")
    fig.colorbar(im02, ax=axs[2])
    im03 = axs[3].contourf(rr, zz, err_perc, cmap="inferno")
    axs[3].plot(FW[:,0], FW[:,1], 'w', lw=1)
    axs[3].set_aspect('equal')
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




