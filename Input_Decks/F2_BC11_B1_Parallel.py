#!/usr/local/python
#

#############################################################################
# This file is simulating the second and third bend magnets in the FACET BC11
# chicane.



from __future__ import print_function #Python 2.7 compatibility
import sys
import os
#sys.path.insert(0, '/Users/brendan/Documents/FACET/SRW/SRW-light/env/work/srw_python') # To find the SRW python libraries.
sys.path.insert(0, '/scratch/brendan/SRW/SRW/env/work/srw_python') # To find
# the SRW python libraries.
# Add the path to the SRW_Parallel library (one directory up)
file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = '/'.join(file_dir.split('/')[:-1])
sys.path.insert(0, file_dir)

# set the backend for matplot lib
import matplotlib
# matplotlib.use("TkAgg")


from srwlib import *
from SRW_Split_Wavefront_Mesh import *
from SRW_Utilities import *

import matplotlib.pyplot as plt
import numpy as np
import time
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()

t0 = time.time()


########################## SETUP THE SIM ###########
def F2_BC11_B1(_Nx):
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
    B0 			    = 0.566604 # Magnetic field strength in Tesla.
    L_Bend 		    = 0.204 # Length of the dipoles in meters.
    L_edge 		    = 0.05 # length of the field edge in meters.
    entry_drift     = 0.3 # Entry and exit drifts, in meters


    # Beam transport simulation parameters.
    npTraj 			= 2**14 # Number of points to compute the trajectory. 2**14
    ctStart 		= 0.0 # start point for tracking simulation, in meters
    ctEnd 			= 1.0*L_Bend + 2.0*L_edge + 2.0*entry_drift
    # end point for tracking simulation, in meters

    photon_lam 	    = 0.65e-6 # Photon wavelength in [m]


    # Wavefront parameters
    # Wavefront mesh parameters
    Nx 			    = _Nx
    Ny 			    = Nx
    B1_phys_edge    = entry_drift + 1.6*1*L_edge# The
    # physical upstream edge of B1 (i.e. where the field has just become flat.)
    zSrCalc 	    = B1_phys_edge + 0.204 + 0.60 + 0.289 # Distance from sim
    # start to
    # calc SR. [m]
    xMiddle		    = 0.0 # middle of window in X to calc SR [m]
    xWidth 		    = 0.040*2.0 # width of x window. [m]
    yMiddle 	    = 0.0 # middle of window in Y to calc SR [m]
    yWidth 		    = xWidth # width of y window. [m]

    # SR integration flags.
    use_termin 		= 1 #1 #Use "terminating terms" (i.e. asymptotic expansions
    # at  zStartInteg and zEndInteg) or not (1 or 0 respectively)
    # Precision for SR integration
    srPrec 		    = 0.05 #0.05

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
    magFldCnt = build_single_magnet(B0, L_Bend, L_edge, entry_drift)
    # magFldCnt = build_two_magnets(B0, L_Bend, L_edge, entry_drift, Bend_sep)


    ################################################################################
    ########################## OPTIMIZE THE SIMULATION #############################
    ################################################################################

    # Run the simulation as built
    print('   Performing initial trajectory calculation ... ', end='')
    partTraj_1 = srwl.CalcPartTraj(partTraj_1, magFldCnt, trajPrecPar)
    print('done.')

    # Set the field strength to get the desired bend angle
    # partTraj_1, magFldCnt = set_mag_strength_by_bend_angle(goal_Bend_Angle, B0,
    #                                            partTraj_1, magFldCnt,
    #                                            L_Bend, L_edge, entry_drift,
    #                                             Bend_sep, trajPrecPar)

    # Set the initial particle offset and angle to make the offset and angle
    #  zero. You want to use the first edge, but it has a lot of background from
    # 'creation' of the particle.
    partTraj_1 = set_initial_offset_and_angle(partTraj_1, magFldCnt,
                                              trajPrecPar, -1)


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


    return wfr1, magFldCnt, arPrecPar, partTraj_1, elecBeam_1


if __name__ == '__main__':

    nx = 2**10
    filename = 'F2_BC11_B1_' + str(nx)

    # Prepare the simulation
    wfr, magFldCnt, arPrecPar, partTraj_1, elecBeam_1 = F2_BC11_B1(nx)

    # Perform the simulation.
    t0 = time.time()
    time_str = 'SR Calculation started at ' + time.ctime() + \
               '. \n'
    print(time_str, end='')

    # # copy the resulting wavefront for safe keeping
    wfr1 = deepcopy(wfr)
    srwl.CalcElecFieldSR(wfr1, 0, magFldCnt, arPrecPar)

    # wfr1 = CalcElecFieldGaussianMPI(wfr, magFldCnt, arPrecPar)
    # Save the wavefront to a file.

    dump_srw_wavefront(filename, wfr1)

    time_str = "Run time: %.2f seconds." % (time.time() - t0)
    print(time_str)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------


    wfr1 = load_srw_wavefront(filename)

    t0 = time.time()
    time_str = 'SRW Physical Optics Calculation started at ' + time.ctime() + \
               '. \n'
    print(time_str, end='')

    wfr_out = deepcopy(wfr1)
    p_dist          = 1.0*0.912
    focal_length    = 1.0*0.050 # in meters
    prop_distance   = 1.0*0.0575 # in meters


    paramsAper  = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
    paramsLens  = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
    paramsDrift = [0, 0, 1., 1, 0, 1., 1., 1., 1., 0, 0, 0]

    a_drift = SRWLOptC(
        [SRWLOptA(_shape = 'c', _ap_or_ob = 'a', _Dx = 0.2),
         SRWLOptL(focal_length, focal_length),
         SRWLOptD(prop_distance)],
        [paramsAper, paramsLens, paramsDrift])
    srwl.PropagElecField(wfr_out, a_drift)

    time_str = "Run time: %.2f seconds." % (time.time() - t0)
    print(time_str)

    # plot_SRW_intensity(wfr1, 1)
    plot_SRW_intensity(wfr_out, 1)

    plot_two_SRW_intensity(wfr1, wfr_out, "Input WF", "Prop. WF", 3)

    aWfr = wfr_out
    II = convert_Efield_to_intensity(aWfr)
    to_plot = II[:, 3*nx // 4]
    y = np.linspace(aWfr.mesh.yStart, aWfr.mesh.yFin, aWfr.mesh.ny)
    y_fwhm = 2.0*y[np.abs(to_plot - to_plot.max()/2.0).argmin()]

    plt.close(13)
    plt.figure(13)
    plt.plot(y, to_plot)
    plt.xlabel('y [mm]', fontsize=20)
    plt.ylabel('Intensity [arb]', fontsize=20)
    plt.tight_layout()

    ##########################################
    # Convolve the single particle wavefront with a beam.

    beam_energy_temp = elecBeam_1.partStatMom1.get_E()
    gamma = elecBeam_1.partStatMom1.get_E() / 0.511e-3

    elecBeam_1.from_Twiss(_Iavg=0.5, _e=beam_energy_temp, _sig_e=0.0,
                          _emit_x=(5e-6 / gamma),
                          _beta_x=1.0,
                          _alpha_x=(5e-6 / gamma) * 1.0,
                          _eta_x=0,
                          _eta_x_pr=0,
                          _emit_y=(5e-6 / gamma),
                          _beta_y=1.0,
                          _alpha_y=(5e-6 / gamma) * 1.0,
                          _eta_y=0,
                          _eta_y_pr=0)



    convWfr = deepcopy(wfr_out)
    xMin = 1e3 * convWfr.mesh.xStart
    xMax = 1e3 * convWfr.mesh.xFin
    yMin = 1e3 * convWfr.mesh.yStart
    yMax = 1e3 * convWfr.mesh.yFin

    convWfr.partBeam = elecBeam_1
    arI2_temp = array('f', [0] * convWfr.mesh.nx * convWfr.mesh.ny)
    srwl.CalcIntFromElecField(arI2_temp, convWfr, 6, 1, 3,
                              convWfr.mesh.eStart, 0, 0)
    C_conv = np.reshape(arI2_temp, [convWfr.mesh.nx, convWfr.mesh.ny])

    plt.close(14)
    plt.figure(14)
    plt.imshow(C_conv, extent=[xMin, xMax, yMin, yMax])
    plt.gca().set_aspect((xMax - xMin) / (yMax - yMin))
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    plt.clim([0, np.max(C_conv)])
    plt.title("Beam Convolved SRW Intensity", fontsize=20)
    plt.set_cmap('jet')
    plt.tight_layout()


    to_plot = C_conv[:, 3*nx // 4]
    y = np.linspace(convWfr.mesh.yStart, convWfr.mesh.yFin, convWfr.mesh.ny)
    y_fwhm2 = 2.0*y[np.abs(to_plot - to_plot.max()/2.0).argmin()]

