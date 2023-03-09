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

    # Run the simulation for the first edge.
    B2B3_first_edge = F2_Single_Magnet_Multiple_Color_Sim(Nx=2**10,
                                  goal_Bend_Angle=-.105 * 180 / np.pi,
                                  meshZ=B2B3_first_edge_to_window,
                                  ph_lam=0.60e-6)
    # Run the SRW calculation
    B2B3_first_edge.run_SR_calculation()
    B2B3_first_edge.propagate_wavefront_through_window()
    B2B3_first_edge.resize_wavefront()


    # Run the simulation for the second edge.
    B2B3_secon_edge = F2_Single_Magnet_Multiple_Color_Sim(Nx=2**10,
                                  goal_Bend_Angle=.105 * 180 / np.pi,
                                  meshZ=B2B3_secon_edge_to_window,
                                  ph_lam=0.60e-6)
    # Run the SRW calculation
    B2B3_secon_edge.run_SR_calculation()
    B2B3_secon_edge.propagate_wavefront_through_window()
    B2B3_secon_edge.resize_wavefront()

    # Combine the two wavefronts
    wfr = deepcopy(B2B3_secon_edge.wfr)
    wfr.addE(B2B3_first_edge.wfr)
    plot_SRW_intensity(wfr, title=sim_title)






