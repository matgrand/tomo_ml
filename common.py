# file with common parameters for the project
import numpy as np
π = 3.141592653589793

GSIZE = 55 #16 #55 # size of the grid (32)
ISIZE = 68 # vdi + vdc + vde + hor1

# Simulated data parameters 
L = 1.101600170135498 # [m] length of the grid in the r/x direction (square grid)
R0 = 1.449199914932251 # [m] grid start in the r/x direction
R_FW = 0.4905 # [m] first wall radius
Z0 = -0.5508000254631042 # [m] grid start in the z/y direction
R1, Z1 = R0+L, Z0+L # [m] grid ends in the r/x and z/y direction
RM, ZM = 0.5*(R0+R1), 0.5*(Z0+Z1) # [m] grid center in the x/r and z direction
# calculated constants
R = np.linspace(R0, R1, GSIZE)
Z = np.linspace(Z0, Z1, GSIZE)
assert np.isclose(R1-R0, Z1-Z0), "grid must be square"
GSPAC = L/GSIZE # [m] grid spacing
RR, ZZ = np.meshgrid(R, Z) # create a grid of R and Z values
# RRL, ZZL = RR[::KHRES, ::KHRES], ZZ[::KHRES, ::KHRES] # create low resolution grid
RZ = np.stack((RR, ZZ), axis=-1) # create a grid of R and Z values
FW = np.array([[RM+R_FW*np.cos(θ), ZM+R_FW*np.sin(θ)] for θ in np.linspace(0, 2*π, 100)]) # [m] first wall

DS_NVDI, DS_NVDC, DS_NVDE, DS_NHOR = (17, 16, 16, 19) # number of rays for each SXR subdivision
DS_SXR_SPLITS = (0, DS_NVDI, DS_NVDI+DS_NVDC, DS_NVDI+DS_NVDC+DS_NVDE, DS_NVDI+DS_NVDC+DS_NVDE+DS_NHOR) # splits for the SXR data


## RFX SXR Parameters (all aproximated)


# lognormal parameters for the max emissivity, fitted from the data (shape, loc, scale)
MAX_EMISS_LOGNORM_PARAMS = (0.8669316172599792, -2.115141144810439, 171.3672332763672) 


# RFX REAL CLEAN DATASET SXR (tot 92) SUBDIVISIONS (my data interpretation)
VDI_INTERVAL = (32, 49) 
VDC_INTERVAL = (16, 32)
VDE_INTERVAL = (0, 16) 
HOR1_INTERVAL = (49, 68)
HOR2_INTERVAL = (68, 92)

# VDI, SXR Vertical Internal 
VDI_SPAN_ANGLE = 0.32004687510760094 + 0.0349 
VDI_START_ANGLE = π/2 - VDI_SPAN_ANGLE/2
VDI_PINHOLE_POS = (1.72 + 0.01, -0.73 +0.01) 
VDI_NRAYS = 19 # number of rays (paper)
VDI_TO_KEEP = np.arange(1, 18) # rays to keep (data) 17 (could be 0->17, 2->19) #

# VDC, SXR Vertical Central
VDC_SPAN_ANGLE = 0.7859524793194145 -0.0175 -0.0175
VDC_START_ANGLE = π/2 - VDC_SPAN_ANGLE/2
VDC_PINHOLE_POS = (2.0, -0.64 +0.01-0.0057)  #(2.0-0.01+0.0057, -0.64 +0.01-0.0057) 
VDC_NRAYS = 19 # number of rays (paper)
VDC_TO_KEEP = np.arange(2, 18) # rays to keep (data) 16 (could be 2->18)

# VDE, SXR Vertical External
VDE_SPAN_ANGLE = 0.9004511925214301 -0.0524 -0.0873 -0.0175
VDE_START_ANGLE = π/2 - VDE_SPAN_ANGLE/2
VDE_PINHOLE_POS = (2.27 -0.02, -0.46-0.1-0.015) 
VDE_NRAYS = 19 # number of rays (paper)
VDE_TO_KEEP = np.arange(1, 17) # rays to keep (data) 16 (could be 0->16, 2->18, 3->19)

# HOR, SXR Horizontal, double
HOR_SPAN_ANGLE = 1.7 -0.0175
HOR_START_ANGLE = π - HOR_SPAN_ANGLE/2
HOR_PINHOLE_POS = (2.53, 0.0)
HOR_NRAYS = 21 # number of rays (paper 24)

HOR1_TO_KEEP = np.arange(2, 21) # rays to keep (data) 19 (may be different)
#HOR2_TO_KEEP = np.arange(0, 24) # rays to keep (data) all of them, 24 # NOTE: impossible with 21 rays, ignore HOR2

RFX_SXR_NAMES = ("VDI", "VDC", "VDE", "HOR")
RFX_SXR_NAMES_2_IDXS = {"VDI":0, "VDC":1, "VDE":2, "HOR":3}
RFX_SXR_IDXS_2_NAMES = {0:"VDI", 1:"VDC", 2:"VDE", 3:"HOR"}
RFX_SXR_COLORS = ('b', 'g', 'r', 'y')
RFX_SXR_INTERVALS = (VDI_INTERVAL, VDC_INTERVAL, VDE_INTERVAL, HOR1_INTERVAL)
RFX_SXR_NRAYS = (VDI_NRAYS, VDC_NRAYS, VDE_NRAYS, HOR_NRAYS)
RFX_SXR_STARTS = (VDI_START_ANGLE, VDC_START_ANGLE, VDE_START_ANGLE, HOR_START_ANGLE)
RFX_SXR_SPANS = (VDI_SPAN_ANGLE, VDC_SPAN_ANGLE, VDE_SPAN_ANGLE, HOR_SPAN_ANGLE)
RFX_SXR_PINHOLES = (VDI_PINHOLE_POS, VDC_PINHOLE_POS, VDE_PINHOLE_POS, HOR_PINHOLE_POS)
RFX_SXR_TO_KEEP = (VDI_TO_KEEP, VDC_TO_KEEP, VDE_TO_KEEP, HOR1_TO_KEEP)

RFX_SXR_LEN = sum([i1-i0 for i0, i1 in RFX_SXR_INTERVALS]) # total number of los
RFX_COMBINED_INTERVALS = sum([list(range(i0, i1)) for i0, i1 in RFX_SXR_INTERVALS], []) # use it to reorder the SXRs in the dataset
