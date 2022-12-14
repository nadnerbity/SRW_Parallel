#!/usr/local/python
#

#############################################################################
# This file is simulating the second and third bend magnets in the FACET BC11
# chicane.



from __future__ import print_function #Python 2.7 compatibility
import sys
#sys.path.insert(0, '/Users/brendan/Documents/FACET/SRW/SRW-light/env/work/srw_python') # To find the SRW python libraries.
sys.path.insert(0, '/home/brendan/SRW/env/work/srw_python') # To find the SRW python libraries.



# set the backend for matplot lib
import matplotlib
# matplotlib.use("TkAgg")


from srwlib import *
from Split_SRW_Wavefront_Mesh import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py
from scipy.optimize import curve_fit
# Some matplotlib settings.
plt.close('all')
plt.ion()

t0 = time.time()

################################################################################


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

########################## SETUP THE SIM ###########
def F2_BC11_B2_and_B3():
    """
    # Setup the simulation for the B2 and B3 magnets in BC11. This is a
    simulation for testing purposes. SRW documentation will help decypher
    what all these variables mean.
    :return:
    """

    # Beam Parameters.
    beam_energy 	= 0.330 # Beam energy in GeV
    x_initial_1 	= 0.0 # Initial x offset in meters
    xp_initial_1 	= 0.0 # initial xprime in rad
    z_initial_1 	= 0.0 # In meters

    # Magnet parameters
    goal_Bend_Angle = 0.105*180/np.pi # In degrees, should be about 6 degrees for
    #  BC11
    B0 			    = 0.6 # Magnetic field strength in Tesla.
    L_Bend 		    = 0.204 # Length of the dipoles in meters.
    L_edge 		    = 0.05 # length of the field edge in meters.
    entry_drift     = 0.3 # Entry and exit drifts, in meters
    Bend_sep        = 0.83 - 1.6*2*L_edge # Distance between the magnet edges if
    # they had zero length edges. Subtract off L_edge because the
    # measurements/Lucretia files dont' include edges in distance. The 1.6 is a
    # fudge factor for in simulation length vs the defined length. The
    # distribution in the documentation appears to be wrong, so I can't
    # analytically reproduce this.

    # Beam transport simulation parameters.
    npTraj 			= 2**14 # Number of points to compute the trajectory. 2**14
    ctStart 		= 0.0 # start point for tracking simulation, in meters
    ctEnd 			= 2.0*L_Bend + 4.0*L_edge + 2.0*entry_drift + Bend_sep
    # end point for tracking simulation, in meters

    photon_lam 	    = 0.65e-6 # Photon wavelength in [m]


    # Wavefront parameters
    # Wavefront mesh parameters
    Nx 			    = 2**7
    Ny 			    = Nx
    B3_phys_edge    = entry_drift + 1.6*3*L_edge + L_Bend + Bend_sep # The
    # physical edge of B3 (i.e. where the field has just become flat.)
    zSrCalc 	    = B3_phys_edge + 0.8209 # Distance from sim start to calc SR. [m]
    xMiddle		    = 0.0 # middle of window in X to calc SR [m]
    xWidth 		    = 0.04 # width of x window. [m]
    yMiddle 	    = 0.00 # middle of window in Y to calc SR [m]
    yWidth 		    = xWidth # width of y window. [m]

    # SR integration flags.
    use_termin 		= 1 #1 #Use "terminating terms" (i.e. asymptotic expansions
    # at  zStartInteg and zEndInteg) or not (1 or 0 respectively)
    # Precision for SR integration
    srPrec 		    = 0.01 #0.01

    ################################################################################
    ################### DERIVED SIMULATION PARAMETERS ##############################
    ################################################################################

    #####################################################
    # A simple electron beam.  Should be just one particle.
    # You're going to use this beam to see if you've built up
    # the magnetic field correctly.

    # Build the first particle trajectory.  This is used to verify the trajectory
    # through the magnets is what you expect.
    part_1 = SRWLParticle()
    part_1.x = x_initial_1
    part_1.y = 0.0
    part_1.xp = xp_initial_1
    part_1.yp = 0.0
    part_1.z = z_initial_1
    part_1.gamma = beam_energy/0.51099890221e-03
    part_1.relE0 = 1 #Electron Rest Mass
    part_1.nq = -1 #Electron Charge

    #**********************Trajectory structure, where the results will be stored
    partTraj_1 = SRWLPrtTrj()
    partTraj_1.partInitCond = part_1
    partTraj_1.allocate(npTraj, True)
    partTraj_1.ctStart = ctStart #Start Time for the calculation
    partTraj_1.ctEnd = ctEnd

    trajPrecPar = [1] #General Precision parameters for Trajectory calculation


    #**********************Build the magnetic field container
    # magFldCnt = build_single_magnet(B0, L_Bend, L_edge, entry_drift)
    magFldCnt = build_two_magnets(B0, L_Bend, L_edge, entry_drift, Bend_sep)

    ################################################################################
    ########################## OPTIMIZE THE SIMULATION #############################
    ################################################################################

    # Run the simulation as built
    print('   Performing initial trajectory calculation ... ', end='')
    partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)
    print('done.')

    # Set the field strength to get the desired bend angle
    partTraj_1, magFldCnt = set_mag_strength_by_bend_angle(goal_Bend_Angle, B0,
                                               partTraj_1, magFldCnt,
                                               L_Bend, L_edge, entry_drift,
                                                Bend_sep, trajPrecPar)

    # Set the initial particle offset and angle to make the offset and angle
    #  zero. You want to use the first edge, but it has a lot of background from
    # 'creation' of the particle.
    partTraj_1 = set_initial_offset_and_angle(partTraj_1, magFldCnt,
                                              trajPrecPar)


    ################################################################################
    ##################### Setup the wavefront calculations #########################
    ################################################################################

    # Parameters that are derived form those above to feed into the script.
    # You shouldn't have to change these.
    photon_e = 4.135e-15 * 299792458.0 / photon_lam # convert wavelength to eV

    #####################################################
    # Set up an electron beam faking a single particle
    # This is what is used to generate the SR.

    elecBeam_1 = SRWLPartBeam()
    elecBeam_1.Iavg = 0.5 #Average Current [A]
    elecBeam_1.partStatMom1.x = partTraj_1.partInitCond.x
    elecBeam_1.partStatMom1.y = partTraj_1.partInitCond.y
    elecBeam_1.partStatMom1.z = partTraj_1.partInitCond.z
    elecBeam_1.partStatMom1.xp = partTraj_1.partInitCond.xp
    elecBeam_1.partStatMom1.yp = partTraj_1.partInitCond.yp
    elecBeam_1.partStatMom1.gamma = beam_energy/0.51099890221e-03 #Relative Energy

    ############################################
    #***********Precision Parameters for SR calculation
    meth = 2 #SR calculation method: 0- "manual", 1- "auto-undulator",
    # 2- "auto-wiggler"
    relPrec = srPrec #relative precision
    zStartInteg = ctStart #longitudinal position to start integration (
    # effective if < zEndInteg)
    zEndInteg = ctEnd #longitudinal position to finish integration (
    # effective if
    #  > zStartInteg)
    npTraj = npTraj #Number of points for trajectory calculation
    useTermin = use_termin #1 #Use "terminating terms" (i.e. asymptotic expansions at zStartInteg and zEndInteg) or not (1 or 0 respectively)
    sampFactNxNyForProp = -1 #sampling factor for adjusting nx, ny (effective if > 0)
    arPrecPar = [meth, relPrec, zStartInteg, zEndInteg, npTraj, useTermin, 0]

    #*********** Wavefront data placeholder
    wfr1 = SRWLWfr() #For spectrum vs photon energy
    wfr1.allocate(1, Nx, Ny) #Numbers of points vs Photon Energy, Horizontal and Vertical Positions
    wfr1.mesh.zStart    = zSrCalc #Longitudinal Position [m] from Center of Straight Section at which SR has to be calculated
    wfr1.mesh.eStart 	= photon_e #Initial Photon Energy [eV]
    wfr1.mesh.eFin 		= photon_e #Final Photon Energy [eV]
    wfr1.mesh.xStart 	= xMiddle - xWidth/2.0 #Initial Horizontal Position [m]
    wfr1.mesh.xFin 		= xMiddle + xWidth/2.0 #Final Horizontal Position [m]
    wfr1.mesh.yStart 	= yMiddle - yWidth/2.0 #Initial Vertical Position [m]
    wfr1.mesh.yFin 		= yMiddle + yWidth/2.0 #Final Vertical Position [m]
    wfr1.partBeam 		= elecBeam_1


    return wfr1, magFldCnt, arPrecPar


if __name__ == '__main__':

    # Prepare the simulation
    wfr, magFldCnt, arPrecPar = F2_BC11_B2_and_B3()

    # copy the resulting wavefront for safe keeping
    wfr1 = deepcopy(wfr)

    # Run vanilla SRW
    start_time = time.time()
    srwl.CalcElecFieldSR(wfr1, 0, magFldCnt, arPrecPar)
    print('------ %s seconds to execute single processor SRW -----' % (
            time.time() - start_time))

    # Perform the same simulation but using N processors.
    wfr2 = CalcElecFieldGaussianMPI(wfr, magFldCnt, arPrecPar)

    # Extract the single simulation intensity pattern.
    arI1 = array('f', [0] * wfr1.mesh.nx * wfr1.mesh.ny)
    srwl.CalcIntFromElecField(arI1, wfr1, 6, 0, 3, wfr1.mesh.eStart, 0, 0)
    Single_Proc = np.reshape(arI1, [wfr1.mesh.nx, wfr1.mesh.ny])

    # Extract the MPI simulation intensity pattern.
    arI1 = array('f', [0] * wfr2.mesh.nx * wfr2.mesh.ny)
    srwl.CalcIntFromElecField(arI1, wfr2, 6, 0, 3, wfr2.mesh.eStart, 0, 0)
    Multi_Proc = np.reshape(arI1, [wfr2.mesh.nx, wfr2.mesh.ny])

    # Plot the result for comparison.
    plt.figure(1, facecolor='w')
    plt.subplot(1,2,1)
    plt.imshow(Single_Proc)
    plt.title('Single Processor')

    plt.subplot(1,2,2)
    plt.imshow(Multi_Proc)
    plt.title('Multiple Processors')