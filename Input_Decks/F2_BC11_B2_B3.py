#!/usr/local/python
#

#############################################################################
# This file file defines a class for a single magnet in the FACET-II dogleg
# and BC11 chicane. This file simulates the B2 and B3 magnets in the BC11
# chicane


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
import itertools
from F2_Single_Magnet_Sim import *
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()



if __name__ == '__main__':

    # Name for plots of this simulation
    # sim_title = "BC11_B2_and_B2"
    sim_title = __file__

    # Create the simulation
    first_edge_to_window = 1.58 # in meters
    secon_edge_to_window = 0.83  # in meters

    ###########
    # Create the simulation
    NN = 2 ** 10
    wavelength = 400
    Lbend = 0.204
    Ledge = 0.05
    beamenergy = 0.330
    Ne = 18
    # p = (537.0e-9, 563.0e-9) # With 25 nm filter
    p = (400.0e-9, 700.0e-9) # Camera only.
    windowToLen = 0.215
    windowApp = 0.038

    a_sim = F2_Single_Magnet_Multiple_Color_Sim(Nx=NN,
                                                goal_Bend_Angle=
                                                1 * .105 * 180 /
                                                np.pi,
                                                meshZ=first_edge_to_window,
                                                ph_lam=wavelength * 1e-9,
                                                L_bend=Lbend,
                                                L_edge=Ledge,
                                                beam_energy=beamenergy,
                                                Ne=Ne,
                                                p=p)
    a_sim.run_SR_calculation()
    a_sim.propagate_wavefront_through_window(appOne=windowApp,
                                             windowToLens=windowToLen)
    a_sim.resize_wavefront()

    b_sim = F2_Single_Magnet_Multiple_Color_Sim(Nx=NN,
                                                goal_Bend_Angle=
                                                -1 * .105 * 180 /
                                                np.pi,
                                                meshZ=secon_edge_to_window,
                                                ph_lam=wavelength * 1e-9,
                                                L_bend=Lbend,
                                                L_edge=Ledge,
                                                beam_energy=beamenergy,
                                                Ne=Ne,
                                                p=p)
    b_sim.run_SR_calculation()
    b_sim.propagate_wavefront_through_window(appOne=windowApp,
                                             windowToLens=windowToLen)
    b_sim.resize_wavefront()

    # b_sim.match_wavefront_mesh_dimensions(a_sim.wfr)
    a_sim.add_wavefront(b_sim.wfr)

    # plot_SRW_intensity(a_sim.wfr, title="One of Many Colors", fig_num=2, N=1)
    plot_SRW_fluence(a_sim.wfr, title=sim_title)

    # write_sim_to_disc(a_sim, sim_title)


    # colors = np.linspace(400e-9, 800e-9, 10)
    # mags = [[first_edge_to_window, -1.], [secon_edge_to_window, 1.0]]
    # sim_list = list(itertools.product(range(len(colors)), mags))
    # NN = 2**10
    #
    # comb2 = F2_Single_Magnet_Multiple_Color_Sim(Nx=NN,
    #                                            goal_Bend_Angle=.105 * 180 / np.pi,
    #                                            meshZ=secon_edge_to_window,
    #                                            ph_lam=0.60e-6)
    # # Ensure the mesh has the correct number of wavelengths
    # comb2.build_wavefront_mesh(Ne=len(colors), p=(colors[0], colors[-1]))
    #
    # for L in sim_list:
    #     a_sim = F2_Single_Magnet_Single_Color_Sim(Nx=NN,
    #                                            goal_Bend_Angle=
    #                                            L[1][1]*.105 * 180 /
    #                                            np.pi,
    #                                            meshZ=L[1][0],
    #                                            ph_lam=colors[L[0]])
    #     a_sim.run_SR_calculation()
    #     a_sim.propagate_wavefront_through_window(windowToLens=0.215)
    #     a_sim.resize_wavefront()
    #     comb2.add_specific_color_wavefront(a_sim.wfr, L[0])
    #
    #
    # comb2.match_wavefront_mesh_dimensions(a_sim.wfr)
    #
    # plot_SRW_intensity(comb2.wfr, title="Single colors stacked into a multi")
    #
    # what3, lol3 = comb2.lineout(xOrY=0, N=int(a_sim.wfr.mesh.nx/2))
    # plt.close(235)
    # plt.figure(235, facecolor='w')
    # plt.plot(what3, lol3, 'bo--')
    # plt.xlim([-500e-6, 500e-6])
    # plt.xlabel('X [m]', fontsize=16)
    # plt.ylabel('Intensity [?]', fontsize=16)
    # plt.title('Ne = ' + str(comb2.wfr.mesh.ne)
    #           + ', ' + str(colors[0]) + ', ' + str(colors[-1]) +
    #           ', Nx = ' + str(comb2.wfr.mesh.nx))


