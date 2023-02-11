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
sys.path.insert(0, '/scratch/brendan/SRW_Parallel/SRW_Parallel')
sys.path.insert(0, '/scratch/brendan/SRW_Parallel/SRW_Parallel/PhysicalOptics')



# set the backend for matplot lib
# import matplotlib
# matplotlib.use("TkAgg")


from srwlib import *
from SRW_Split_Wavefront_Mesh import *
from SRW_Utilities import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import FourierOptics as FO
from scipy.optimize import curve_fit
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()

def fit_func(x, a, x0, c, b):
    return np.angle(np.exp((1j/c)*(x-x0)**2)*np.exp(1j*b))



if __name__ == '__main__':

    # Load the wavefront to a file.
    filename = 'F2_BC11_B2_and_B3_Nx_1024'
    # filename = 'F2_BC11_B2_and_B3_Small_Nx_4096'
    # filename = 'Robbie_After_SR_Calc_4096'
    wfr = load_srw_wavefront(filename)
    plot_SRW_intensity(wfr, 1)

    EM = convert_srw_linear_fields_to_matrix_fields(wfr)
    Ex = EM[:, :, 0] + 1j * EM[:, :, 1]
    x = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.ny)


    to_fit_and_plot = np.angle(Ex[:, wfr.mesh.nx//4])
    to_weight = np.abs(Ex[:, wfr.mesh.nx//4])**4
    pguess = [0.5,
              0,
              3.0e-7,
              to_fit_and_plot[to_fit_and_plot.size // 2]]
    bguess = [[0, -1, 0.1*pguess[2], 1.1*pguess[3]],
              [1, 1, 10.0*pguess[2], 0.9*pguess[3]]]
    popt, pcov = curve_fit(fit_func, x,
                    to_fit_and_plot,
                    pguess,
                    bounds = bguess,
                    sigma = 100/to_weight,
                    ftol = 1e-15,
                    xtol=1e-15)
    fitOut = fit_func(x, popt[0], popt[1], popt[2], popt[3])

    # fitOut = fit_func(x, 1, 0, 2e-7, -2)

    plt.close(2)
    plt.figure(2)
    plt.plot(x, to_fit_and_plot, 'bo')
    plt.plot(x, fitOut, 'rx')
    plt.xlim([-0.005, 0.005])
    plt.legend(['Data', 'Fit'])

    # plt.figure(3)
    # plt.figure(3)
    # plt.plot(x, to_fit_and_plot - fitOut, 'bo')