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
    secon_edge_to_window = 2.0*0.76  # in meters

    color = 500e-9
    NN = 2**8

    Nc = 2**3
    lc = 31e-9 # Critical wavelength
    # C = np.linspace(1, 100, Nc) * lc
    C = np.logspace(-0.5, 3, Nc)
    A = np.zeros((Nc,1))
    E = np.zeros((Nc, 1))
    EE = np.zeros((Nc, 1))

    for i in range(Nc):
        a_sim = F2_Single_Magnet_Single_Color_Sim(Nx=NN,
                                                  goal_Bend_Angle=
                                                  1 * .105 * 180 /
                                                  np.pi,
                                                  meshZ=secon_edge_to_window,
                                                  ph_lam=C[i]*lc)

        a_sim.run_SR_calculation()

        I = convert_Efield_to_intensity(a_sim.wfr)
        A[i] = I[NN//2, 3*NN//4]
        E[i] = I[NN // 2, NN // 2]
        EE[i] = I[NN//2-10:NN//2+10, NN//2-10:NN//2+10].mean()

    plot_SRW_intensity(a_sim.wfr, title="Single color, single edge")

    plt.figure(234)
    plt.loglog(1/C, A, 'bo-')
    plt.loglog(1/C, EE, 'rx-')
    plt.xlabel('$\lambda$ [nm]')
    temp = plt.xticks()
    plt.xticks(ticks=temp[0], labels=[str(1e9*lc/x) for x in temp[0]])
    plt.xlim([0.5 / C[-1], 2 / C[-0]])
    plt.ylabel('Intensity [arb]', fontsize=18)
    plt.legend(('Synchrotron Radiation','Edge Radiation'), fontsize=18)

