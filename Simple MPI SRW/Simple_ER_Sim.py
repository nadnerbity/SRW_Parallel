#!/usr/local/python
#

#############################################################################
# This file is simulating the dogleg at FACET 2.
# v 0.01 Got the single particle trajectory correct
# v 0.10 Functions written to set field strength based on desired bend angle
# v 0.20 Added ability to fit Gaussian to output radiation width
# v 0.30 Added ability to convolve single particle SR with a beam.
# Based on SRWLIB_Example01.py
#############################################################################



from __future__ import print_function #Python 2.7 compatibility
import sys
#sys.path.insert(0, '/Users/brendan/Documents/FACET/SRW/SRW-light/env/work/srw_python') # To find the SRW python libraries.
sys.path.insert(0, '/home/brendan/SRW/env/work/srw_python') # To find the SRW python libraries.

from srwlib import *
import os
import numpy as np
import time


# This function builds a SRW magnetic field container containing two
# identical magnets.
def build_two_magnets(B0, L_Bend, L_edge, entry_drift, Bend_sep):
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

def run_simple_er_simulation(nx_in):
    ########################## SETUP THE SIM ##################

    # Beam Parameters.
    beam_energy 	= 0.330 # Beam energy in GeV
    x_initial_1 	= 0.04769510598771189 # Initial x offset in meters
    xp_initial_1 	= -0.10500002118040946 # initial xprime in rad
    z_initial_1 	= 0.0 # In meters

    # Magnet parameters
    goal_Bend_Angle = 0.105*180/np.pi # In degrees, should be about 6 degrees for
    #  BC11
    B0 			    = 0.5666041013230535 # Magnetic field strength in Tesla.
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
    npTraj 			= 2**13 # Number of points to compute the trajectory. 2**14
    ctStart 		= 0.0 # start point for tracking simulation, in meters
    ctEnd 			= 2.0*L_Bend + 4.0*L_edge + 2.0*entry_drift + Bend_sep
    # end point for tracking simulation, in meters

    photon_lam 	    = 0.65e-6 # Photon wavelength in [m]


    # Wavefront parameters
    # Wavefront mesh parameters
    Nx = nx_in
    Ny = Nx
    B3_phys_edge    = entry_drift + 1.6*3*L_edge + L_Bend + Bend_sep # The
    # physical edge of B3 (i.e. where the field has just become flat.)
    zSrCalc 	    = B3_phys_edge + 0.8209 # Distance from sim start to calc SR. [m]
    xMiddle		    = 0.0 # middle of window in X to calc SR [m]
    xWidth 		    = 0.040 # width of x window. [m]
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
    magFldCnt = build_two_magnets(B0, L_Bend, L_edge, entry_drift, Bend_sep)

    ################################################################################
    ########################## OPTIMIZE THE SIMULATION #############################
    ################################################################################

    # Run the simulation as built
    # print('   Performing initial trajectory calculation ... ', end='')
    partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)
    # print('done.')

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


    ######################## Perform SR calculation ################################
    # t0 = time.time()
    # time_str = 'SR Calculation started at ' + time.ctime() + '. \n'
    # print(time_str, end='')
    # print('Running simulation with nx=ny=', nx_in)

    # print('   Performing wavefront calculation ... ', end='')
    srwl.CalcElecFieldSR(wfr1, 0, magFldCnt, arPrecPar)
    # print('done.')
    #
    # time_str = "Run time: %.2f seconds." % (time.time() - t0)
    # print(time_str)
    return wfr1

if __name__ == '__main__':
    wfr = run_simple_er_simulation(2**11)