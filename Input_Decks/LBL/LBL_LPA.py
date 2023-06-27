#!/usr/local/python
#

#############################################################################
# This file is simulating the second and third bend magnets in the FACET BC11
# chicane.



from __future__ import print_function #Python 2.7 compatibility
import sys
#sys.path.insert(0, '/Users/brendan/Documents/FACET/SRW/SRW-light/env/work/srw_python') # To find the SRW python libraries.
sys.path.insert(0, '/scratch/brendan/SRW/SRW/env/work/srw_python') # To find
# the SRW python libraries.
sys.path.insert(0, '/home/boshea/SRW_Parallel/')


# set the backend for matplot lib
import matplotlib
# matplotlib.use("TkAgg")


from srwlib import *
# from SRW_Split_Wavefront_Mesh import *
from SRW_Utilities import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()

t0 = time.time()


########################## SETUP THE SIM ###########
def F2_BC11_B2_and_B3():
    """
    # Setup the simulation for the B2 and B3 magnets in BC11. This is a
    simulation for testing purposes. SRW documentation will help decypher
    what all these variables mean.
    :return:
    """

    # Beam Parameters.
    beam_energy 	= 0.100 # Beam energy in GeV
    x_initial_1 	= 0.0 # Initial x offset in meters
    xp_initial_1 	= 0.0 # initial xprime in rad
    z_initial_1 	= 0.0 # In meters

    # Magnet parameters
    goal_Bend_Angle = 15.0 # In degrees, should be about 15 degrees for LBL
    B0 			    = 0.9 # Magnetic field strength in Tesla.
    L_Bend 		    = 0.097 # Length of the dipoles in meters.
    L_edge 		    = 0.015 # length of the field edge in meters.
    entry_drift     = 0.3 # Entry and exit drifts, in meters
    Bend_sep        = 0.1 # Distance between the magnet edges if
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

    photon_lam 	    = 0.500e-6 # Photon wavelength in [m]


    # Wavefront parameters
    # Wavefront mesh parameters
    Nx 			    = 2**10
    Ny 			    = Nx
    zSrCalc 	    = 1.0 # Distance from sim start to calc SR. [m]
    xMiddle		    = -0.0025 # middle of window in X to calc SR [m]
    xWidth 		    = 0.010 # width of x window. [m]
    yMiddle 	    = 0.0 # middle of window in Y to calc SR [m]
    yWidth 		    = xWidth # width of y window. [m]

    # SR integration flags.
    use_termin 		= 1 #1 #Use "terminating terms" (i.e. asymptotic expansions
    # at  zStartInteg and zEndInteg) or not (1 or 0 respectively)
    # Precision for SR integration
    srPrec 		    = 0.05 #0.01

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
    magFldCnt = build_two_magnets_dogleg(B0, L_Bend, L_edge, entry_drift,
                                       Bend_sep)

    ################################################################################
    ########################## OPTIMIZE THE SIMULATION #############################
    ################################################################################

    # Run the simulation as built
    print('   Performing initial trajectory calculation ... ', end='')
    partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)
    print('done.')

    # Set the field strength to get the desired bend angle
    partTraj_1, magFldCnt = set_mag_strength_by_bend_angle_dogleg(
        goal_Bend_Angle,B0,partTraj_1, magFldCnt,L_Bend, L_edge, entry_drift,
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


    return wfr1, magFldCnt, arPrecPar, partTraj_1


if __name__ == '__main__':

    # Prepare the simulation
    wfr, magFldCnt, arPrecPar, partTraj_1 = F2_BC11_B2_and_B3()

    # copy the resulting wavefront for safe keeping
    wfr1 = deepcopy(wfr)

    # Run vanilla SRW
    start_time = time.time()
    srwl.CalcElecFieldSR(wfr1, 0, magFldCnt, arPrecPar)
    print('------ %s seconds to execute single processor SRW -----' % (
            time.time() - start_time))

    # Extract the single simulation intensity pattern.
    arI1 = array('f', [0] * wfr1.mesh.nx * wfr1.mesh.ny)
    srwl.CalcIntFromElecField(arI1, wfr1, 6, 0, 3, wfr1.mesh.eStart, 0, 0)
    Single_Proc = np.reshape(arI1, [wfr1.mesh.nx, wfr1.mesh.ny])


    # # Plot the result for comparison.
    # plt.figure(1, facecolor='w')
    # # plt.subplot(1,2,1)
    # plt.imshow(Single_Proc)
    # plt.title('LBL HTU FEL numbers')

    plot_SRW_intensity(wfr1, fig_num=2, title='LBL HTU FEL numbers')
