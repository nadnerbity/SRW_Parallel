
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
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
plt.close('all')
plt.ion()





def build_single_magnet(B0, L_Bend, L_edge, entry_drift):
    """
    # This function builds a SRW magnetic field container containing a single
    # magnet.

    :param B0: Magnetic field strength [Tesla]
    :param L_Bend: Length of the flattop region of the bend magnetic field [
    meters]
    :param L_edge: Length of the magnetic field edges [meters]
    :param entry_drift: Drift from simulation start to the first magnet edge
    [meters]
    :return: magFldCont - SRW magnetic field container for the magnets
    """
    bend1 = SRWLMagFldM()
    bend1.m = 1  # 1 defines a dipole
    bend1.G = -B0  # Field strength of the bend in Tesla, since it is a dipole
    bend1.Leff = L_Bend  # Effective dipole length, in meters.
    bend1.Ledge = L_edge  # Edge length in meters.the ID)

    z1 = entry_drift + L_Bend /2.0 + L_edge

    # Offsets for all the magnetic fields in the magFldCnt.
    bendy = [bend1]
    xcID = [0.0]
    ycID = [0.0]
    zcID = [z1]

    # Put everything together.  These are the two fields.
    magFldCnt = SRWLMagFldC(bendy, array('d', xcID), array('d', ycID),
                            array('d', zcID))
    return magFldCnt


def build_two_magnets(B0, L_Bend, L_edge, entry_drift, Bend_sep):

    """
    # This function builds a SRW magnetic field container containing two
    # identical magnets for simulating edge radiation interference.

    :param B0: Magnetic field strength [Tesla]
    :param L_Bend: Length of the flattop region of the bend magnetic field [
    meters]
    :param L_edge: Length of the magnetic field edges [meters]
    :param entry_drift: Drift from simulation start to the first magnet edge
    [meters]
    :param Bend_sep: Separation between the middle magnets, not including
    edges [meters]
    :return: magFldCont - SRW magnetic field container for the magnets
    """
    bend1 = SRWLMagFldM()
    bend1.m = 1  # 1 defines a dipole
    bend1.G = -B0  # Field strength of the bend in Tesla, since it is a dipole
    bend1.Leff = L_Bend  # Effective dipole length, in meters.
    bend1.Ledge = L_edge  # Edge length in meters.the ID)

    bend2 = SRWLMagFldM()
    bend2.m = 1  # 1 defines a dipole
    bend2.G = -B0  # Field strength of the bend in Tesla, since it is a dipole
    bend2.Leff = L_Bend  # Effective dipole length, in meters.
    bend2.Ledge = L_edge  # Edge length in meters.the ID)

    z1 = entry_drift + L_Bend /2.0 + L_edge
    z2 = z1 + L_Bend + Bend_sep + L_edge

    # Offsets for all the magnetic fields in the magFldCnt.
    bendy = [bend1, bend2]
    xcID = [0.0]*2
    ycID = [0.0]*2
    zcID = [z1, z2]

    # Put everything together.  These are the two fields.
    magFldCnt = SRWLMagFldC(bendy, array('d', xcID), array('d', ycID),
                            array('d', zcID))
    return magFldCnt



def set_mag_strength_by_bend_angle(goal_Bend_Angle, B0, partTraj_1,
                                   magFldCnt, L_Bend, L_edge, entry_drift,
                                   Bend_sep, trajPrecPar):
    """
    When the entrance and exit edge fields change length, the total bend angle
    will change. This function sets the field strength such that the bend angle
     is the user input value.

    # This function is written for pairs of bend magnets and sets the angle and
    # offset to zero in the middle of the two magnets.

    :param goal_Bend_Angle: The desired bend angle, in degrees
    :param B0: Magnetic field of the flat region of the bend. [Tesla]
    :param partTraj_1: An SRW particle trajectory variable
    :param magFldCnt: The magnetic field container with the magnetic system.
    :param L_Bend: Length of the flattop of the bend magnet [meters]
    :param L_edge: Length of the edge of the magnetic field [meters]
    :param entry_drift: Drift from simulation start to the first magnet edge
    [meters]
    :param Bend_sep: Separation between the middle magnets, not including
    edges [meters]
    :param trajPrecPar: Precision parameter for the trajectory calculations.
    :return: partTraj_1, magFldCnt
    """
    print('   Setting the magnetic field strength to match bend angle ... ')


    # Looks like it takes three iterations to converge.
    N = 3
    for i in range(N):
        # Extract the current bend angle.
        # partTraj_1.np//2 is what make sthe function set the bend angle in
        # the middle of the magFldCnt
        curr_Bend_Angle = partTraj_1.arXp[partTraj_1.np//2] * 180 / pi

        delta_Theta = (curr_Bend_Angle - goal_Bend_Angle) / curr_Bend_Angle
        B0 = B0 * (1 - delta_Theta)

        # Update the magnetic field to the new value.
        # magFldCnt = build_single_magnet(B0, L_Bend, L_edge, entry_drift)
        magFldCnt = build_two_magnets(B0, L_Bend, L_edge, entry_drift, Bend_sep)

        # Run the updated particle trajectory
        partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)

    print('Final magnetic field strength: ' + str(B0) + ' [T]')

    return partTraj_1, magFldCnt


def set_simulation_length(partTraj_1, magFldCnt, Np = 2):

    """
    # To make the length of simulation symmetric in magnets with large bend
    # angle, like the dogleg magnets, you need to know the total bend angle. They
    #  bend so much that the particle doesn't travel as far in Z because it is
    # deflected far enough in Y to make a difference. This function runs the
    # simulation for the input ctEnd and returns the difference between the
    # desired ctEnd and the distance in Z actually traveled by the particle.

    :param partTraj_1: SRW particle trajectory class instance
    :param magFldCnt:  SRW magnetic field class instance
    :return: SRW particle trajectory class with the correct ctEnd to get the
    desired z
    """
    print('   Setting the simulation length ... ')
    ctEndGoal = partTraj_1.ctEnd
    N = 4  # Number of times to try and get the simulation length correct.

    for k in range(N):
        # z_diff = ctEndGoal - partTraj_1.arZ[-1]
        z_diff = ctEndGoal - partTraj_1.arZ[partTraj_1.np//Np]
        partTraj_1.ctEnd = partTraj_1.ctEnd + z_diff
        partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)

    print('Final simulation length: ' + str(partTraj_1.ctEnd) + ' [m]')

    return partTraj_1


def set_initial_offset_and_angle(partTraj_1, magFldCnt, trajPrecPar, Np = None):
    """
    # I setup all simulations so that the wavefront plane is on axis. This
    means that when I change the magnetic field strength I need to make sure
    the beam is on axis in the middle fo the magnets with zero angle. This
    function sets the input offset and angle such that after the first
    magnet the beam is on axis and has no angle due to the bend

    :param partTraj_1: SRW Particle trajectory class
    :param magFldCnt:  SRW magnetic field class
    :param trajPrecPar: Precision to simulation the trajectory
    :return: partTraj_1 with no offset and angle in the middle of the two bends
    """
    if Np is None:
        Np = partTraj_1.np//2

    print('   Setting the initial offset and angle ... ')
    partTraj_1.partInitCond.x = partTraj_1.arX[Np]
    partTraj_1.partInitCond.xp = -partTraj_1.arXp[Np]

    # Run the updated particle trajectory
    partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)

    # Do it again to get the offset very small. Again, this should only need to
    # be done once but there are precision parameters I don't fully understand.

    partTraj_1.partInitCond.x = partTraj_1.partInitCond.x - partTraj_1.arX[Np]
    partTraj_1.partInitCond.xp = partTraj_1.partInitCond.xp - \
        partTraj_1.arXp[Np]

    # Run the updated particle trajectory
    partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)

    print("Angle at entrance " + str(partTraj_1.arXp[0] * 180 / pi) + \
                                                                    " degrees")
    print("Angle in middle " + str(partTraj_1.arXp[Np] * 180 /
                                   pi) + \
                                                                    " degrees")
    print("Offset in middle " + str(partTraj_1.arX[Np] * 1000.0)
          + \
          " [mm]")


    return partTraj_1


def dump_srw_wavefront(filename, wfr_in):
    """
    Dump SRW wavefront to a pickle file

    :param filename: name of the file to dump to (string)
    :param wfr: SRW wavefront to dump
    :return: nothing
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        filename = filename + '.pkl'
        f = open(filename, 'wb')
        pickle.dump(wfr_in, f)
        f.close()


def load_srw_wavefront(filename):
    """
    Load and SRW wavefront from a pickle file

    :param filename: name of the file to load from (string)
    :return: SRW wavefront that was loaded
    """
    filename = filename + '.pkl'
    f = open(filename, 'rb')
    wfr = pickle.load(f)
    f.close()
    return wfr


def convert_Efield_to_intensity(wfr1):
    """
    # SRW stores fields in a linear array, but you want to plot intensity as an
    # image. SRW stores the data as Er1, Ei1, Er2, Ei2, Er3, Ei3...
    # Which means it stores the fields alternating between real and imaginary.

    # This does the same thing as:
    arI1 = array('f', [0] * wfr1.mesh.nx * wfr1.mesh.ny)
    srwl.CalcIntFromElecField(arI1, wfr1, 6, 0, 3, wfr1.mesh.eStart, 0, 0)
    II = np.reshape(arI1, [wfr1.mesh.nx, wfr1.mesh.ny])

    :param wfr1: Wavefront to extract the intensity from.
    :return: Intensity of the wavefront
    """
    Nx = wfr1.mesh.nx
    Ny = wfr1.mesh.ny

    II = np.zeros((Ny, Nx))
    for i in range(Nx):
        for j in range(Ny):
            II[j, i] = wfr1.arEx[2 * (Nx * j + i)] ** 2 + \
                       wfr1.arEx[2 * (Nx * j + i) + 1] ** 2 + \
                       wfr1.arEy[2 * (Nx * j + i)] ** 2 + \
                       wfr1.arEy[2 * (Nx * j + i) + 1] ** 2

    # # # This shows how the indexing works
    # Nx = 4
    # Ny = 3
    # III = np.zeros((Ny, Nx))
    # for i in range(Nx):
    #     for j in range(Ny):
    #         # print(str( 2 * (Nx * j + i) ))
    #         III[j, i] = 2 * (Nx * j + i)

    return II


def convert_srw_linear_fields_to_matrix_fields(wfr1):
    """
    # Convert the SRW linear data type to a matrix (my brain works in images). It
    #  takes in a wavefront and returns 4 matrices, real(Ex), imaginary(Ex),
    # real(Ey) and imaginary(Ey). The output is a numpy array of size (Ny, Nx, 4)

    :param wfr1: SRW wavefront to convert to a matrix
    :return: Numpy matrix of the field (nx x ny x 4)
    """
    Nx = wfr1.mesh.nx
    Ny = wfr1.mesh.ny

    II = np.zeros((Ny, Nx, 4))
    for i in range(Nx):
        for j in range(Ny):
            II[j, i, 0] = wfr1.arEx[2 * (Nx * j + i)]
            II[j, i, 1] = wfr1.arEx[2 * (Nx * j + i) + 1]
            II[j, i, 2] = wfr1.arEy[2 * (Nx * j + i)]
            II[j, i, 3] = wfr1.arEy[2 * (Nx * j + i) + 1]

    return II


def convert_matrix_fields_to_srw_linear_fields(II, wfr1):
    """
    Convert an (ny x nx x 4) matrix of electric fields into an SRW style set
    of Ex and Ey fields.

    :param II: Electric fields in format (ny x nx x 4)
    See 'convert_srw_linear_fields_to_matrix_fields'
    :param wfr1: SRW wavefront to deposit the fields onto
    :return: SRW wavefront with the fields deposited on it
    """
    Nx = wfr1.mesh.nx
    Ny = wfr1.mesh.ny

    if (II.shape[0] != Ny or II.shape[1] != Nx):
        raise IndexError('Matrix field shape and wavefront shape do not '
                         'agree. Matrix field shape is ' + str(II.shape) \
                         + ' srw mesh is (' + str(Ny) +',' + str(Nx) + ')'
                         )

    for i in range(Nx):
        for j in range(Ny):
            wfr1.arEx[2 * (Nx * j + i)] = II[j, i, 0]
            wfr1.arEx[2 * (Nx * j + i) + 1] = II[j, i, 1]
            wfr1.arEy[2 * (Nx * j + i)] = II[j, i, 2]
            wfr1.arEy[2 * (Nx * j + i) + 1] = II[j, i, 3]

    return wfr1


def plot_SRW_intensity(wfr1, fig_num=2):
    Nx = wfr1.mesh.nx
    Ny = wfr1.mesh.ny
    xMin = 1e3 * wfr1.mesh.xStart
    xMax = 1e3 * wfr1.mesh.xFin
    yMin = 1e3 * wfr1.mesh.yStart
    yMax = 1e3 * wfr1.mesh.yFin
    # xgv = np.linspace(xMin, xMax, Nx)
    # ygv = np.linspace(yMin, yMax, Ny)

    # Extract the single particle intensity
    arI1 = array('f', [0] * wfr1.mesh.nx * wfr1.mesh.ny)
    srwl.CalcIntFromElecField(arI1, wfr1, 6, 0, 3, wfr1.mesh.eStart, 0, 0)
    B = np.reshape(arI1, [Ny, Nx])

    plt.figure(fig_num, facecolor='w')

    plt.imshow(B, extent=[xMin, xMax, yMin, yMax])
    plt.gca().set_aspect((xMax - xMin) / (yMax - yMin))
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    plt.clim([0, np.max(B)])
    plt.title("SRW Intensity", fontsize=20)
    plt.set_cmap('jet')
    plt.tight_layout()

def plot_two_SRW_intensity(wfr1, wfr2, title1="Input 1", title2="Input 2",
                           fig_num=2):
    Nx = wfr1.mesh.nx
    Ny = wfr1.mesh.ny
    xMin1 = 1e3 * wfr1.mesh.xStart
    xMax1 = 1e3 * wfr1.mesh.xFin
    yMin1 = 1e3 * wfr1.mesh.yStart
    yMax1 = 1e3 * wfr1.mesh.yFin

    # Extract the single particle intensity
    arI1 = array('f', [0] * wfr1.mesh.nx * wfr1.mesh.ny)
    srwl.CalcIntFromElecField(arI1, wfr1, 6, 0, 3, wfr1.mesh.eStart, 0, 0)
    B = np.reshape(arI1, [Ny, Nx])

    Nx = wfr2.mesh.nx
    Ny = wfr2.mesh.ny
    xMin2 = 1e3 * wfr2.mesh.xStart
    xMax2 = 1e3 * wfr2.mesh.xFin
    yMin2 = 1e3 * wfr2.mesh.yStart
    yMax2 = 1e3 * wfr2.mesh.yFin

    # Extract the single particle intensity
    arI1 = array('f', [0] * wfr2.mesh.nx * wfr2.mesh.ny)
    srwl.CalcIntFromElecField(arI1, wfr2, 6, 0, 3, wfr2.mesh.eStart, 0, 0)
    C = np.reshape(arI1, [Ny, Nx])

    plt.figure(fig_num, facecolor='w')

    plt.subplot(1,2,1)
    plt.imshow(B, extent=[xMin1, xMax1, yMin1, yMax1])
    plt.gca().set_aspect((xMax1 - xMin1) / (yMax1 - yMin1))
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    plt.clim([0, np.max(B)])
    plt.title(title1, fontsize=20)


    plt.subplot(1,2,2)
    plt.imshow(C, extent=[xMin2, xMax2, yMin2, yMax2])
    plt.gca().set_aspect((xMax2 - xMin2) / (yMax2 - yMin2))
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    plt.clim([0, np.max(C)])
    plt.title(title2, fontsize=20)
    plt.tight_layout()
