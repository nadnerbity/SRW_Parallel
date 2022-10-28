
#############################################################################
# This file if for splitting up an SRW mesh into many small meshes so they
# can be simulated independently. This will hopefully allow a simulation to
# be parallelized.
#############################################################################

# This set of functions divides a 'full sized' SRW wavefront mesh (the x and
# y part) into pieces so they can be calculated separately and essentially
# parallelized. The x and y part of an SRW grid is defined by:
# Nx, Ny : the number of grid points in X and Y.
# xStart, yStart : The start of the x and y grid, in meters
# xFin, yFin : The finish of the x and y grid, in meters
#
# For ease of updating, Brendan uses xWidth and xMiddle (similar for y) to
# set xStart and xFin. xWidth is the total width of the grid (in meters) and
# xMiddle is the center of the grid (in meters).
# xStart = -xWidth/2.0 + xMiddle
# xFin = xWidth/2.0 + xMiddle


import sys
sys.path.insert(0, '/home/brendan/SRW/env/work/srw_python') # To find the SRW python libraries.
from srwlib import *

class subSRWLWfr(SRWLWfr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgrid = SRWsubgrid()

class SRWsubgrid():
    def __init__(self):
        self.gridNum = -1
        self.iMin = 0
        self.iMax = 0
        self.jMin = 0
        self.jMax = 0

    def set_i_and_i_ranges(self):
        pass

# This function returns the indices for the subgrid based on processor number.
def subgrid_by_number(h, Nx, Ny, gx, gy):
    if h < 0 or h >= gx*gy:
        raise ValueError('Subgrid number must be between [0, gx*gy-1]')

    iMin = (h - gx * (h//gx)) * (Nx/gx)
    iMax = iMin + Nx/gx - 1
    jMin = (h//gx) * Ny/gy
    jMax = jMin + Ny/gy - 1

    return [iMin, iMax, jMin, jMax]


# Create a subgrid wavefront based on the user supplied main wavefront
    # Need to check here if wfr_in.mesh.nx > 1 and wfr_in.mesh.ny > 1, otherwise
    # will get divide by zero when calculating either of the deltas

def sub_wavefront_from_main_wavefront(h, wfr_in, gx, gy):
    indices = subgrid_by_number(h, wfr_in.mesh.nx, wfr_in.mesh.ny, gx, gy)

    # Create a new wavefront mesh
    wfrt = SRWLWfr()
    # Build the wavefront meshgrid
    wfrt.allocate(1, wfr_in.mesh.nx // gx, wfr_in.mesh.nx // gy)
    # Assign synchrotron radiation calculation parameters
    wfrt.mesh.zStart = wfr_in.mesh.zStart  # Longitudinal Position [m] from Center of
    # Straight Section at which SR has to be calculated
    wfrt.mesh.eStart = wfr_in.mesh.eStart  # Initial Photon Energy [eV]
    wfrt.mesh.eFin = wfr_in.mesh.eFin  # Final Photon Energy [eV]

    # Calculate the grid spacing on the input grid
    deltaX = (wfr_in.mesh.xFin - wfr_in.mesh.xStart) / (wfr_in.mesh.nx - 1)
    deltaY = (wfr_in.mesh.yFin - wfr_in.mesh.yStart) / (wfr_in.mesh.ny - 0)


    # Update the Start and Fin for both axes
    wfrt.mesh.xStart = indices[0] * deltaX + wfr_in.mesh.xStart
    wfrt.mesh.xFin = indices[1] * deltaX + wfr_in.mesh.xStart
    wfrt.mesh.yStart = indices[2] * deltaY + wfr_in.mesh.yStart
    wfrt.mesh.yFin = indices[3] * deltaY + wfr_in.mesh.yStart

    return wfrt




# Nx = 2**3
# Ny = 2**2
# gx = 2
# gy = 2
#
# for i in range(gx*gy):
#     print(subgrid_by_number(i, Nx, Ny, gx, gy))