#!/usr/local/python
#

#############################################################################
# This file file defines a class for a single magnet in the FACET-II dogleg
# and BC11 chicane. They are the same magnets, so I'm building a class to
# handle doing simulations magnet, but magnet so I can combine them as I see
# fit after the fact.
# A basic SRW comes in 4 parts:
# 1) Build a magnetic field container
# 2) Use (1) to calculate a particle trajectory
# 3) Build a wavefront object to deposit fields onto
# 4) Use the particle trajectory to calculate fields and deposit them on the
# wavefront object from (3)


import sys
import os
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


class F2_Single_Magnet_Sim:

    # Magnet parameters --------------------------------------------------------
    # The design bend angle, from theory or simulation. In degrees.
    goal_Bend_Angle = 0.105 * 180 / np.pi

    # Magnetic field strength in Tesla.
    B0 = 0.6

    # Length of the 'hard' edge of the dipoles in meters.
    L_Bend = 0.204

    # Length of the field edge in meters.
    L_edge = 0.05

    # Entry and exit drifts, in meters. The simulation can't start in a
    # magnetic field.
    entry_drift = 0.3

    # Distance between the magnet edges if they had zero length edges.
    # Subtract off L_edge because the measurements/Lucretia files don't
    # include edges in distance. The 1.6 is a fudge factor for in simulation
    # length vs the defined length.
    Bend_sep = 0.93 - 1.6 * 2 * L_edge

    # Initial beam parameters
    # Beam Parameters.
    beam_energy 	= 0.330 # Beam energy in GeV
    x_initial_1 	= 0.0 # Initial x offset in meters
    xp_initial_1 	= 0.0 # initial xprime in rad
    z_initial_1 	= 0.0 # In meters

    def __init__(self):
        self.magFldCnt = self.build_single_magnet()

    def build_single_magnet(self):
        """
        # This function builds a SRW magnetic field container containing a single
        # magnet.

        :return: magFldCont - SRW magnetic field container for the magnets
        """
        bend1 = SRWLMagFldM()
        bend1.m = 1  # 1 defines a dipole
        # Field strength of the bend in Tesla, since it is a dipole
        bend1.G = self.B0
        bend1.Leff = self.L_Bend  # Effective dipole length, in meters.
        bend1.Ledge = self.L_edge  # Edge length in meters.the ID)

        z1 = self.entry_drift + self.L_Bend / 2.0 + self.L_edge

        # Offsets for all the magnetic fields in the magFldCnt.
        bendy = [bend1]
        xcID = [0.0]
        ycID = [0.0]
        zcID = [z1]

        # Put everything together.  These are the two fields.
        magFldCnt = SRWLMagFldC(bendy, array('d', xcID), array('d', ycID),
                                array('d', zcID))
        return magFldCnt





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

