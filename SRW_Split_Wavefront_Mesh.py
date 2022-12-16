
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
sys.path.insert(0, '/scratch/brendan/SRW/SRW/env/work/srw_python') # To find
# the SRW python libraries.
from srwlib import *
import pickle
from mpi4py import MPI
import time


def CalcElecFieldGaussianMPI(wfr_main, magFldCnt, arPrecPar):
    """
    A function to divide up an SRW simulation and perform it on multiple
    cores. This is used in place of the SRW method CalcElecFieldSR.

    :param wfr_main: SRW wavefront class that contains the simulation.
    :param magFldCnt: SRW magentic field container class for the simulation
    :param arPrecPar: The precision parameters for SRW (see SRW documentation)
    :return: SRW wavefront class with the simulation results.
    """

    try:
        MPI.COMM_WORLD
    except:
        print('MPI does not appear to be running.')
        print('Returning input wavefront unchanged.')
        return wfr_main

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()

    # Define the way to divide up the simulations space. It turns out that
    # you wrote a general function, but for the meth =2 simulation type in
    # SRW, the simulation is not embarrassingly parallelizable, so it is best
    #  to divide up the x space.
    gx = world
    gy = 1
    Ns = gx*gy

    if Ns != world:
        raise ValueError('gx*gy must equal the number of processors')

    if rank == 0:
        start_time = time.time()

    start = time.time()

    wfr_empty = deepcopy(wfr_main)
    # Split the wavefront up into pieces and run the simulation
    wfr_sub = sub_wavefront_from_main_wavefront(rank, wfr_empty, gx, gy)
    # Run the simulation
    srwl.CalcElecFieldSR(wfr_sub, 0, magFldCnt, arPrecPar)
    print('Time for one MPI Simulation', time.time() - start)

    # Hold here until all processors are done.
    comm.Barrier()

    if rank != 0:
        comm.send(wfr_sub, dest=0, tag=rank)
    elif rank == 0:
        wfr_main = copy_sub_mesh_to_main_mesh(rank, wfr_main, gx, gy, wfr_sub)
        # Collect the data from the other nodes
        start = time.time()
        for i in range(1, world):
            # print('in the receive loop')
            temp = comm.recv(source=i, tag=i)
            wfr_main = copy_sub_mesh_to_main_mesh(i, wfr_main, gx, gy, temp)

        print('Data collection time', time.time() - start)


        print('------ %s seconds to execute MPI version of SRW -----' % (
                time.time() - start_time))

    # Hold here until all processors are done.
    comm.Barrier()


    return wfr_main


def CalcElecFieldGaussianSerial(wfr_main, g_beam_in):

    """
    This function is mainly for testing. It uses one processor and runs the
    functions used to divide up an SRW wavefront mesh so that it can be
    simulated on multiple cores to reduce simulation time.

    It uses the SRW Gaussian beam functions because those are fast. I'm
    checking that the divide + simulate + recombine functions work, not scaling.

    :param wfr_main: Wavefront used to define the simulation space
    :param g_beam_in: SRW class with the gaussian photon beam parameters
    :return: wfr_main with the complete simulation deposited on it.
    """

    # Hard code some things while I figure out how to do it correctly but
    # want to keep moving on the things I know.
    gx = 2
    gy = 2
    Ns = gx*gy

    for s in range(Ns):
        # Copy the original wavefront to make sure you don't mess up the
        # original
        wfr_empty = deepcopy(wfr_main)
        # Split the wavefront up into pieces based on on step number 's'
        wfr_sub = sub_wavefront_from_main_wavefront(s, wfr_empty, gx, gy)
        # Deposit a Gaussian beam on the pieces of the wavefront mesh
        srwl.CalcElecFieldGaussian(wfr_sub, g_beam_in, [0])
        # Copy the pieces back to the main mesh.
        wfr_main = copy_sub_mesh_to_main_mesh(s, wfr_main, gx, gy, wfr_sub)

    return wfr_main


def subgrid_by_number(h, Nx, Ny, gx, gy):
    """
    # SRW Mesh class contains a mesh and the physical limits of the mesh.
    This function determines where on the full mesh a sub mesh sits. From
    these indices and full mesh parameters, the physical extent of the
    submesh can be calculated.

    # See sub_wavefront_from_main_wavefront for definition of what a subgrid is.

    :param h: sub mesh to generate
    :param Nx: Number of grid points in in x of the full grid (not the subgrid)
    :param Ny: Number of grid points in in y of the full grid (not the subgrid)
    :param gx: number of subdivisions in x
    :param gy: number of subdivisions in y
    :return:
    """
    if h < 0 or h >= gx*gy:
        raise ValueError('Sub mesh number must be between [0, gx*gy-1]')

    iMin = (h - gx * (h//gx)) * (Nx/gx)
    iMax = iMin + Nx/gx - 1
    jMin = (h//gx) * Ny/gy
    jMax = jMin + Ny/gy - 1

    return [iMin, iMax, jMin, jMax]


def sub_wavefront_from_main_wavefront(h, wfr_in, gx, gy):
    """
    # Create a sub mesh wavefront based on the user supplied main wavefront
    mesh.
    # It divides the simulation mesh into gx components along the x axis and
    gy components along the y axis. "h" is the sub mesh you're trying to
    generate.

    examples:
    gx=gy=2
    -----------------
    | h = 1 | h = 2 |
    -----------------
    | h = 3 | h = 4 |
    -----------------

    gx = 4 gy = 1
    ---------------------------------
    |       |       |       |       |
    | h = 1 | h = 2 | h = 3 | h = 4 |
    |       |       |       |       |
    ---------------------------------

    gx = 1 gy = 4
    ---------------------------------
    |             h = 1             |
    ---------------------------------
    |             h = 2             |
    ---------------------------------
    |             h = 3             |
    ---------------------------------
    |             h = 4             |
    ---------------------------------


    :param h: sub mesh to generate
    :param wfr_in: full wavefront mesh to generate sub mesh from
    :param gx: number of subdivisions in x
    :param gy: number of subdivisions in y
    :return: wfr_in : SRW wavefront class with the desired sub mesh parameters
    """

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


def copy_sub_mesh_to_main_mesh(h, wfr_main, gx, gy, wfr_sub):
    """
    This function copies the data from the sub mesh to the main mesh that
    the sub mesh was derived from.

    :param h: sub mesh to generate
    :param wfr_in: full wavefront mesh to generate sub mesh from
    :param gx: number of subdivisions in x
    :param gy: number of subdivisions in y
    :param wfr_sub: The SRW wavefront class that contains the data to
    transfer to the main mesh.
    :return: wfr_main with the sub mesh data added
    """

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
            wfr_main.arEx[gamma+1] = wfr_sub.arEx[gammaS+1]
            wfr_main.arEy[gamma] = wfr_sub.arEy[gammaS]
            wfr_main.arEy[gamma+1] = wfr_sub.arEy[gammaS+1]

    return wfr_main

def dump_sub_mesh(h, wfr_sub):
    filename = 'temp_' + str(h) + '.pickle'
    f = open(filename, 'wb')
    pickle.dump(wfr_sub, f)
    f.close()

def load_sub_mesh(h):
    filename = 'temp_' + str(h) + '.pickle'
    f = open(filename, 'rb')
    wfr_sub = pickle.load(f)
    f.close()
    os.remove(filename)
    return wfr_sub


