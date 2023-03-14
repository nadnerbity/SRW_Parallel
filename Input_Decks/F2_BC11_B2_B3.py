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
    sim_title = "B2 and B2"

    # Create the simulation
    B2B3_first_edge_to_window = 1.795 - 0.215 # in meters
    B2B3_secon_edge_to_window = 1.795 - 0.215 - 0.75  # in meters


    colors = [500e-9, 600e-9]
    mags = [[B2B3_first_edge_to_window, -1.], [B2B3_secon_edge_to_window, 1.0]]
    sim_list = list(itertools.product(range(len(colors)), mags))

    comb2 = F2_Single_Magnet_Multiple_Color_Sim(Nx=2 ** 10,
                                               goal_Bend_Angle=.105 * 180 / np.pi,
                                               meshZ=B2B3_secon_edge_to_window,
                                               ph_lam=0.60e-6)

    for L in sim_list:
        a_sim = F2_Single_Magnet_Single_Color_Sim(Nx=2 ** 10,
                                               goal_Bend_Angle=
                                               L[1][1]*.105 * 180 /
                                               np.pi,
                                               meshZ=L[1][0],
                                               ph_lam=colors[L[0]])
        a_sim.run_SR_calculation()
        a_sim.propagate_wavefront_through_window()
        a_sim.resize_wavefront()
        comb2.add_specific_color_wavefront(a_sim.wfr, L[0])

    comb2.wfr.mesh.xStart    = a_sim.wfr.mesh.xStart
    comb2.wfr.mesh.xFin      = a_sim.wfr.mesh.xFin
    comb2.wfr.mesh.yStart    = a_sim.wfr.mesh.yStart
    comb2.wfr.mesh.yFin      = a_sim.wfr.mesh.yFin

    plot_SRW_intensity(comb2.wfr, title="Single colors stacked into a multi")

    # what3, lol3 = comb2.lineout(xOrY=0, N=512)
    # plt.close(235)
    # plt.figure(235, facecolor='w')
    # plt.plot(what, lol, 'bo')
    # plt.plot(what3, lol3, 'rx')
    # plt.legend(['Two colors at once', 'one color at a time'])
    # plt.xlim([-500e-6, 500e-6])


    # # ------------------------------------------------------------------------
    # # Method to do two multicolor simulations at once
    # # Run the simulation for the first edge.
    # # B2B3_first_edge = F2_Single_Magnet_Multiple_Color_Sim(Nx=2**10,
    #                               goal_Bend_Angle=-.105 * 180 / np.pi,
    #                               meshZ=B2B3_first_edge_to_window,
    #                               ph_lam=0.60e-6)
    # # Run the SRW calculation
    # B2B3_first_edge.run_SR_calculation()
    # B2B3_first_edge.propagate_wavefront_through_window()
    # B2B3_first_edge.resize_wavefront()
    #
    #
    # # Run the simulation for the second edge.
    # B2B3_secon_edge = F2_Single_Magnet_Multiple_Color_Sim(Nx=2**10,
    #                               goal_Bend_Angle=.105 * 180 / np.pi,
    #                               meshZ=B2B3_secon_edge_to_window,
    #                               ph_lam=0.60e-6)
    # # Run the SRW calculation
    # B2B3_secon_edge.run_SR_calculation()
    # B2B3_secon_edge.propagate_wavefront_through_window()
    # B2B3_secon_edge.resize_wavefront()
    #
    # # Combine the two wavefronts
    # comb = deepcopy(B2B3_first_edge)
    # comb.add_wavefront(B2B3_secon_edge.wfr)
    # # wfr = deepcopy(B2B3_secon_edge.wfr)
    # # wfr.addE(B2B3_first_edge.wfr)
    # plot_SRW_intensity(comb.wfr, title=sim_title)
    #
    # what, lol = comb.lineout(xOrY=0, N=512)
    # plt.close(234)
    # plt.figure(234, facecolor='w')
    # plt.plot(what, lol)


    # # ------------------------------------------------------------------------
    # # Method to stack many single color siulations into one multi color
    # # simulations
    #
    #
    #
    # B1_500 = F2_Single_Magnet_Single_Color_Sim(Nx=2**10,
    #                               goal_Bend_Angle=-.105 * 180 / np.pi,
    #                               meshZ=B2B3_first_edge_to_window,
    #                               ph_lam=0.50e-6)
    # B1_500.run_SR_calculation()
    # B1_500.propagate_wavefront_through_window()
    # B1_500.resize_wavefront()
    #
    # B1_600 = F2_Single_Magnet_Single_Color_Sim(Nx=2**10,
    #                               goal_Bend_Angle=-.105 * 180 / np.pi,
    #                               meshZ=B2B3_first_edge_to_window,
    #                               ph_lam=0.60e-6)
    # B1_600.run_SR_calculation()
    # B1_600.propagate_wavefront_through_window()
    # B1_600.resize_wavefront()
    #
    # B2_500 = F2_Single_Magnet_Single_Color_Sim(Nx=2 ** 10,
    #                                            goal_Bend_Angle=.105 * 180 / np.pi,
    #                                            meshZ=B2B3_secon_edge_to_window,
    #                                            ph_lam=0.50e-6)
    # B2_500.run_SR_calculation()
    # B2_500.propagate_wavefront_through_window()
    # B2_500.resize_wavefront()
    #
    # B2_600 = F2_Single_Magnet_Single_Color_Sim(Nx=2 ** 10,
    #                                            goal_Bend_Angle=.105 * 180 / np.pi,
    #                                            meshZ=B2B3_secon_edge_to_window,
    #                                            ph_lam=0.60e-6)
    # B2_600.run_SR_calculation()
    # B2_600.propagate_wavefront_through_window()
    # B2_600.resize_wavefront()
    #
    # comb = F2_Single_Magnet_Multiple_Color_Sim(Nx=2 ** 10,
    #                                            goal_Bend_Angle=.105 * 180 / np.pi,
    #                                            meshZ=B2B3_secon_edge_to_window,
    #                                            ph_lam=0.60e-6)
    # comb.add_specific_color_wavefront(B1_500.wfr, 0)
    # comb.add_specific_color_wavefront(B1_600.wfr, 1)
    # comb.add_specific_color_wavefront(B2_500.wfr, 0)
    # comb.add_specific_color_wavefront(B2_600.wfr, 1)
    # comb.wfr.mesh.xStart    = B1_500.wfr.mesh.xStart
    # comb.wfr.mesh.xFin      = B1_500.wfr.mesh.xFin
    # comb.wfr.mesh.yStart    = B1_500.wfr.mesh.yStart
    # comb.wfr.mesh.yFin      = B1_500.wfr.mesh.yFin
    #
    # what2, lol2 = comb.lineout(xOrY=0, N=512)
    # plt.close(234)
    # plt.figure(234, facecolor='w')
    # plt.plot(what, lol, 'bo')
    # plt.plot(what2, lol2, 'rx')
    # plt.legend(['Two colors at once', 'one color at a time'])
    # plt.xlim([-500e-6, 500e-6])