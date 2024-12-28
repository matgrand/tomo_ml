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
FW = np.array([[RM+L/2*np.cos(θ), ZM+L/2*np.sin(θ)] for θ in np.linspace(0, 2*π, 100)]) # [m] first wall

# RFX SXR Parameters (all aproximated)
# VDI, SXR Vertical Internal 
VDI_SPAN_ANGLE = 0.32004687510760094
VDI_START_ANGLE = π/2 - VDI_SPAN_ANGLE/2
VDI_PINHOLE_POS = (1.72, -0.73) 
VDI_NRAYS = 19 # number of rays (paper)
VDI_TO_KEEP = np.arange(3, 19) # rays to keep (data) 16 (could be 2->18)

# VDC, SXR Vertical Central
VDC_SPAN_ANGLE = 0.7859524793194145
VDC_START_ANGLE = π/2 - VDC_SPAN_ANGLE/2
VDC_PINHOLE_POS = (2.0, -0.64) 
VDC_NRAYS = 19 # number of rays (paper)
VDC_TO_KEEP = np.arange(1, 17) # rays to keep (data) 16 (could be 2->18)

# VDE, SXR Vertical External
VDE_SPAN_ANGLE = 0.9004511925214301
VDE_START_ANGLE = π/2 - VDE_SPAN_ANGLE/2
VDE_PINHOLE_POS = (2.27, -0.46) 
VDE_NRAYS = 19 # number of rays (paper)
VDE_TO_KEEP = np.arange(2, 19) # rays to keep (data) 17 (could be 1->18)

# HOR, SXR Horizontal, double
HOR_SPAN_ANGLE = 1.2495650930032198
HOR_START_ANGLE = π - HOR_SPAN_ANGLE/2
HOR_PINHOLE_POS = (2.53, 0.0)
HOR_NRAYS = 21 # number of rays (paper)
HOR1_TO_KEEP = np.arange(1, 20) # rays to keep (data) 19 (may be different)
HOR2_TO_KEEP = np.arange(0, 24) # rays to keep (data) all of them, 24 # NOTE: impossible with 21 rays, ignore HOR2

# RFX REAL CLEAN DATASET SXR (tot 92) SUBDIVISIONS (my data interpretation)
VDI_INTERVAL = (0, 16)
VDC_INTERVAL = (16, 32)
VDE_INTERVAL = (32, 49)
HOR1_INTERVAL = (49, 68)
HOR2_INTERVAL = (68, 92)