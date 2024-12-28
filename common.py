# file with common parameters for the project
GRID_SIZE = 32 # size of the grid (32)
# SXRV_SIZE = 21 #2
# SXRH_SIZE = 23 #23 
INPUT_SIZE = 68 # vdi + vdc + vde + hor1

π = 3.141592653589793

# RFX SXR Parameters (all aproximated)
from numpy import arange

# VDI, SXR Vertical Internal 
VDI_SPAN_ANGLE = 0.32004687510760094
VDI_START_ANGLE = π/2 - VDI_SPAN_ANGLE/2
VDI_PINHOLE_POS = (1.72, -0.73) 
VDI_NRAYS = 19 # number of rays (paper)
VDI_TO_KEEP = arange(3, 19) # rays to keep (data) 16 (could be 2->18)

# VDC, SXR Vertical Central
VDC_SPAN_ANGLE = 0.7859524793194145
VDC_START_ANGLE = π/2 - VDC_SPAN_ANGLE/2
VDC_PINHOLE_POS = (2.0, -0.64) 
VDC_NRAYS = 19 # number of rays (paper)
VDC_TO_KEEP = arange(1, 17) # rays to keep (data) 16 (could be 2->18)

# VDE, SXR Vertical External
VDE_SPAN_ANGLE = 0.9004511925214301
VDE_START_ANGLE = π/2 - VDE_SPAN_ANGLE/2
VDE_PINHOLE_POS = (2.27, -0.46) 
VDE_NRAYS = 19 # number of rays (paper)
VDE_TO_KEEP = arange(2, 19) # rays to keep (data) 17 (could be 1->18)

# HOR, SXR Horizontal, double
HOR_SPAN_ANGLE = 1.2495650930032198
HOR_START_ANGLE = π - HOR_SPAN_ANGLE/2
HOR_PINHOLE_POS = (2.53, 0.0)
HOR_NRAYS = 21 # number of rays (paper)
HOR1_TO_KEEP = arange(1, 20) # rays to keep (data) 19 (may be different)
HOR2_TO_KEEP = arange(0, 24) # rays to keep (data) all of them, 24 # NOTE: impossible with 21 rays, ignore HOR2

# RFX REAL CLEAN DATASET SXR (tot 92) SUBDIVISIONS (my data interpretation)
VDI_INTERVAL = (0, 16)
VDC_INTERVAL = (16, 32)
VDE_INTERVAL = (32, 49)
HOR1_INTERVAL = (49, 68)
HOR2_INTERVAL = (68, 92)