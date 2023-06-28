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
    sim_title = __file__

    # Create the simulation
    first_edge_to_window = 3.16 # in meters
    secon_edge_to_window = 0.76  # in meters
    NN = 2**8
    wavelength = 400
    Lbend = 0.204
    Ledge = 0.05
    beamenergy = 0.330

    a_sim = F2_Single_Magnet_Single_Color_Sim(Nx=NN,
                                              goal_Bend_Angle=
                                              1 * .105 * 180 /
                                              np.pi,
                                              meshZ=first_edge_to_window,
                                              ph_lam=wavelength*1e-9,
                                              L_bend=Lbend,
                                              L_edge=Ledge,
                                              beam_energy = beamenergy)
    a_sim.resize_wavefront(newX=10.e-3, newY=10.0e-3)
    a_sim.run_SR_calculation()

    b_sim = F2_Single_Magnet_Single_Color_Sim(Nx=NN,
                                              goal_Bend_Angle=
                                              1 * .105 * 180 /
                                              np.pi,
                                              meshZ=secon_edge_to_window,
                                              ph_lam=wavelength*1e-9,
                                              L_bend=Lbend,
                                              L_edge=Ledge,
                                              beam_energy = beamenergy)
    b_sim.resize_wavefront(newX=10.e-3, newY=10.0e-3)
    b_sim.run_SR_calculation()


    a_sim.add_wavefront(b_sim.wfr)

    plot_SRW_intensity(a_sim.wfr, title="Single Color")


    # a_sim.dump_wavefront_to_h5(filename=sim_title.split(".")[0] + "_" + str(wavelength) + "_nm.h5")

