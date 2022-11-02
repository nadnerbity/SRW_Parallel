
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



def CalcElecFieldGaussianMPI(wfr_main, g_beam_in):
    # Hard code some things while I figure out how to do it correctly but
    # want to keep moving on the things I know.
    gx = 2
    gy = 2
    Ns = gx*gy

    for s in range(Ns):
        wfr_sub = deepcopy(wfr_main)
        wfr_sub = sub_wavefront_from_main_wavefront(s, wfr_sub, gx, gy)
        srwl.CalcElecFieldGaussian(wfr_sub, g_beam_in, [0])
        wfr_main = copy_sub_mesh_to_main_mesh(s, wfr_main, gx, gy, wfr_sub)

    return wfr_main

class subSRWLWfr(SRWLWfr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgrid = SRWsubgrid()

    def set_i_and_j_ranges(self, subgrid_num):
        if subgrid_num < 0 or subgrid_num >= self.subgrid.gx * self.subgrid.gy:
            raise ValueError('Subgrid number must be between [0, gx*gy-1]')

        self.subgrid.gridNum = subgrid_num

        # I'm going to copy the variables here because otherwise these
        # calculations are inscrutable
        Nx = self.mesh.nx
        Ny = self.mesh.ny
        gx = self.subgrid.gx
        gy = self.subgrid.gy
        h  = self.subgrid.gridNum

        self.subgrid.iMin = (h - gx * (h // gx)) * (Nx / gx)
        self.subgrid.iMax = self.subgrid.iMin + Nx / gx - 1
        self.subgrid.jMin = (h // gx) * Ny / gy
        self.subgrid.jMax = self.subgrid.jMin + Ny / gy - 1

    def set_mesh_ranges(self):
        xWidth = (self.mesh.xFin - self.mesh.xStart)
        yWidth = (self.mesh.yFin - self.mesh.yStart)
        xMiddle = self.mesh.xStart + xWidth / 2.0
        yMiddle = self.mesh.yStart + yWidth / 2.0
        deltaX = xWidth / (self.mesh.nx - 1)
        deltaY = yWidth / (self.mesh.ny - 1)

        # Update the grid physical range parameters
        self.mesh.xStart = self.subgrid.iMin * deltaX + xMiddle
        self.mesh.xFin   = self.subgrid.iMax * deltaX + xMiddle
        self.mesh.yStart = self.subgrid.jMin * deltaY + yMiddle
        self.mesh.yFin   = self.subgrid.jMax * deltaY + yMiddle

        # Update the mesh sizes
        self.mesh.nx = self.mesh.nx // self.subgrid.gx
        self.mesh.ny = self.mesh.ny // self.subgrid.gy

class SRWsubgrid():
    def __init__(self):
        self.gridNum    = -1
        self.iMin       = 0
        self.iMax       = 0
        self.jMin       = 0
        self.jMax       = 0
        self.gx         = 1
        self.gy         = 1


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

    # Calculate the grid spacing on the input grid
    xWidth  = (wfr_in.mesh.xFin - wfr_in.mesh.xStart)
    yWidth  = (wfr_in.mesh.yFin - wfr_in.mesh.yStart)
    deltaX  = xWidth / (wfr_in.mesh.nx - 1)
    deltaY  = yWidth / (wfr_in.mesh.ny - 1)
    xMiddle = (wfr_in.mesh.xFin + wfr_in.mesh.xStart) / 2.0
    yMiddle = (wfr_in.mesh.yFin + wfr_in.mesh.yStart) / 2.0

    # Update the Start and Fin for both axes of the subgrid
    wfr_in.mesh.xFin = wfr_in.mesh.xStart + indices[1] * deltaX
    wfr_in.mesh.xStart = wfr_in.mesh.xStart + indices[0] * deltaX
    wfr_in.mesh.yFin   = wfr_in.mesh.yStart + indices[3] * deltaY
    wfr_in.mesh.yStart = wfr_in.mesh.yStart + indices[2] * deltaY

    wfr_in.mesh.nx      = wfr_in.mesh.nx // gx
    wfr_in.mesh.ny      = wfr_in.mesh.ny // gy
    print("h: " , h, " xStart: ", wfr_in.mesh.xStart, " xFin: ",
        wfr_in.mesh.xFin) # debug
    print("h: " , h, " yStart: ", wfr_in.mesh.yStart, " yFin: ",
        wfr_in.mesh.yFin) # debug
    return wfr_in

# Copy the subgrid fields to the main grid
def copy_sub_mesh_to_main_mesh(h, wfr_main, gx, gy, wfr_sub):

    for k in range(wfr_sub.mesh.nx):
        for m in range(wfr_sub.mesh.ny):
            gamma = 2 * wfr_main.mesh.nx * m + \
                2 * k + \
                2 * wfr_main.mesh.nx * (h//gx) * (wfr_main.mesh.ny//gy - 1) + \
                2 * h * wfr_main.mesh.nx // gx
            gammaS = 2 * wfr_sub.mesh.nx * m + 2 * k
            # print("h: ", h, " m: ", m, " k: ", k, " gamma : " , gamma ,
            # " gammaS:" , gammaS) # debug
            wfr_main.arEx[gamma] = wfr_sub.arEx[gammaS]
            wfr_main.arEy[gamma] = wfr_sub.arEy[gammaS]

    return wfr_main

# Nx = 2**3
# Ny = 2**2
# gx = 2
# gy = 2
#
# for i in range(gx*gy):
#     print(subgrid_by_number(i, Nx, Ny, gx, gy))