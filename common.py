# file with common parameters for the project
import numpy as np

GRID_SIZE = 32 # size of the grid (32)
# SXRV_SIZE = 21 #2
# SXRH_SIZE = 23 #23 
INPUT_SIZE = 68 # vdi + vdc + vde + hor1

π = 3.141592653589793

# Simulated data parameters 1.449199914932251,1.101600170135498
KHRES = 2 # [#] multiplier for the high resolution of the grid
RES = GRID_SIZE*KHRES # [#] high resolution of the grid in pixels (square grid)
L = 1.101600170135498 # [m] length of the grid in the r/x direction (square grid)
R0 = 1.449199914932251 # [m] grid start in the r/x direction
R_FW = 0.4905 # [m] first wall radius
Z0 = -0.5508000254631042 # [m] grid start in the z/y direction
R1, Z1 = R0+L, Z0+L # [m] grid ends in the r/x and z/y direction
RM, ZM = 0.5*(R0+R1), 0.5*(Z0+Z1) # [m] grid center in the x/r and z direction
# calculated constants
R = np.linspace(R0, R1, RES)
Z = np.linspace(Z0, Z1, RES)
assert np.isclose(R1-R0, Z1-Z0), "grid must be square"
δ = L/RES # [m] grid spacing
RRH, ZZH = np.meshgrid(R, Z) # create a grid of R and Z values
RRL, ZZL = RRH[::KHRES, ::KHRES], ZZH[::KHRES, ::KHRES] # create low resolution grid
RZ = np.stack((RRH, ZZH), axis=-1) # create a grid of R and Z values
FW = np.array([[RM+R_FW*np.cos(θ), ZM+R_FW*np.sin(θ)] for θ in np.linspace(0, 2*π, 100)]) # [m] first wall

DS_NVDI, DS_NVDC, DS_NVDE, DS_NHOR = (17, 16, 16, 19) # number of rays for each SXR subdivision
DS_SXR_SPLITS = (0, DS_NVDI, DS_NVDI+DS_NVDC, DS_NVDI+DS_NVDC+DS_NVDE, DS_NVDI+DS_NVDC+DS_NVDE+DS_NHOR) # splits for the SXR data


## RFX SXR Parameters (all aproximated)

# scaling factor for the SXR/EMISS data, only for training convergence purposes
KS = 3000 

# lognormal parameters for the max emissivity, fitted from the data (shape, loc, scale)
MAX_EMISS_LOGNORM_PARAMS = (0.8669316172599792, -2.115141144810439, 171.3672332763672) 



# RFX REAL CLEAN DATASET SXR (tot 92) SUBDIVISIONS (my data interpretation)
VDI_INTERVAL = (32, 49) 
VDC_INTERVAL = (16, 32)
VDE_INTERVAL = (0, 16) 
HOR1_INTERVAL = (49, 68)
HOR2_INTERVAL = (68, 92)

# VDI, SXR Vertical Internal 
VDI_SPAN_ANGLE = 0.32004687510760094
VDI_START_ANGLE = π/2 - VDI_SPAN_ANGLE/2
VDI_PINHOLE_POS = (1.72, -0.73) 
VDI_NRAYS = 19 # number of rays (paper)
VDI_TO_KEEP = np.arange(1, 18) # rays to keep (data) 17 (could be 0->17, 2->19) #

# VDC, SXR Vertical Central
VDC_SPAN_ANGLE = 0.7859524793194145
VDC_START_ANGLE = π/2 - VDC_SPAN_ANGLE/2
VDC_PINHOLE_POS = (2.0, -0.64) 
VDC_NRAYS = 19 # number of rays (paper)
VDC_TO_KEEP = np.arange(2, 18) # rays to keep (data) 16 (could be 2->18)

# VDE, SXR Vertical External
VDE_SPAN_ANGLE = 0.9004511925214301
VDE_START_ANGLE = π/2 - VDE_SPAN_ANGLE/2
VDE_PINHOLE_POS = (2.27, -0.46) 
VDE_NRAYS = 19 # number of rays (paper)
VDE_TO_KEEP = np.arange(1, 17) # rays to keep (data) 16 (could be 0->16, 2->18, 3->19)

# HOR, SXR Horizontal, double
HOR_SPAN_ANGLE = 1.5 #(1.2495650930032198 from paper)
HOR_START_ANGLE = π - HOR_SPAN_ANGLE/2
HOR_PINHOLE_POS = (2.53, 0.0)
HOR_NRAYS = 21 # number of rays (paper 24)

HOR1_TO_KEEP = np.arange(2, 21) # rays to keep (data) 19 (may be different)
#HOR2_TO_KEEP = np.arange(0, 24) # rays to keep (data) all of them, 24 # NOTE: impossible with 21 rays, ignore HOR2
