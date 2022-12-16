
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
    # When the entrance and exit edge fields change length, the total bend angle
    # will change. This function sets the field strength such that the bend angle
    #  is the user input value.

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


def set_simulation_length(partTraj_1, magFldCnt):

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
        z_diff = ctEndGoal - partTraj_1.arZ[partTraj_1.np//2]
        partTraj_1.ctEnd = partTraj_1.ctEnd + z_diff
        partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)

    print('Final simulation length: ' + str(partTraj_1.ctEnd) + ' [m]')

    return partTraj_1


def set_initial_offset_and_angle(partTraj_1, magFldCnt, trajPrecPar):
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
    print('   Setting the initial offset and angle ... ')
    partTraj_1.partInitCond.x = partTraj_1.arX[partTraj_1.np//2]
    partTraj_1.partInitCond.xp = -partTraj_1.arXp[partTraj_1.np//2]

    # Run the updated particle trajectory
    partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)

    # Do it again to get the offset very small. Again, this should only need to
    # be done once but there are precision parameters I don't fully understand.

    partTraj_1.partInitCond.x = partTraj_1.partInitCond.x - partTraj_1.arX[partTraj_1.np//2]
    partTraj_1.partInitCond.xp = partTraj_1.partInitCond.xp - \
        partTraj_1.arXp[partTraj_1.np//2]

    # Run the updated particle trajectory
    partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)

    print("Angle at entrance " + str(partTraj_1.arXp[0] * 180 / pi) + \
                                                                    " degrees")
    print("Angle in middle " + str(partTraj_1.arXp[partTraj_1.np//2] * 180 / pi) + \
                                                                    " degrees")
    print("Offset in middle " + str(partTraj_1.arX[partTraj_1.np//2] * 1000.0)
          + \
          " [mm]")



    return partTraj_1


