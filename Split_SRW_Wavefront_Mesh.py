
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
    gy = 1
    Ns = gx*gy

    for s in range(Ns):
        wfr_sub = deepcopy(wfr_main)
        wfr_sub = sub_wavefront_from_main_wavefront(s, wfr_sub, gx, gy)
        srwl.CalcElecFieldGaussian(wfr_sub, g_beam_in, [0])
        wfr_main = copy_sub_mesh_to_main_mesh(s, wfr_main, gx, gy, wfr_sub)

    return wfr_main

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
    # if wfr_in.mesh.nx or wfr_in.mesh.ny <= 1:
    #     raise ValueError('Input wavefront must have mesh size nx and ny >=1')

    if (wfr_in.mesh.nx % 2) or (wfr_in.mesh.ny % 2) != 0:
        raise ValueError('Input wavefront mesh nx and ny must be even integers')

    indices = subgrid_by_number(h, wfr_in.mesh.nx, wfr_in.mesh.ny, gx, gy)

    # Calculate the grid spacing on the input grid
    deltaX  = (wfr_in.mesh.xFin - wfr_in.mesh.xStart) / (wfr_in.mesh.nx - 1)
    deltaY  = (wfr_in.mesh.yFin - wfr_in.mesh.yStart) / (wfr_in.mesh.ny - 1)

    # Update the Start and Fin for both axes of the subgrid
    wfr_in.mesh.xFin = wfr_in.mesh.xStart + indices[1] * deltaX
    wfr_in.mesh.xStart = wfr_in.mesh.xStart + indices[0] * deltaX
    wfr_in.mesh.yFin   = wfr_in.mesh.yStart + indices[3] * deltaY
    wfr_in.mesh.yStart = wfr_in.mesh.yStart + indices[2] * deltaY
    wfr_in.mesh.nx      = wfr_in.mesh.nx // gx
    wfr_in.mesh.ny      = wfr_in.mesh.ny // gy
    # print("h: " , h, " xStart: ", wfr_in.mesh.xStart, " xFin: ",
    #     wfr_in.mesh.xFin) # debug
    # print("h: " , h, " yStart: ", wfr_in.mesh.yStart, " yFin: ",
    #     wfr_in.mesh.yFin) # debug
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


